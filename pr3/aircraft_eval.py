"""Airplane experiment: image retrieval, vision-language inference, reproducible metrics.
Calls to local Ollama HTTP API are opt-in. Predictions are not simulated.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'data' / 'aircraft_manifest.csv'
IMAGES = ROOT / 'data' / 'images'
OUT = ROOT / 'outputs'
API = 'https://commons.wikimedia.org/w/api.php'
SESSION = requests.Session()
SESSION.headers.update({'User-Agent':'AircraftVisionLLMEvaluation/1.0 (educational data retrieval; see Wikimedia Commons source credits)'})
MODELS = ['qwen2.5vl:3b', 'gemma3:4b']  # 8 GB Mac: one model at a time
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')
OPTIONS = {'temperature':0, 'seed':42, 'num_ctx':2048, 'num_predict':100}
MAX_IMAGE_DIM = 512
PROMPT = ("Look ONLY at the attached image. Decide whether the aircraft shown is MILITARY or CIVILIAN "
          "by the aircraft's pictured operator and primary operational use, NOT just jet/propeller, size, or presence of weapons. "
          "Military includes fighters, reconnaissance, patrol, tanker, training, and military transport; "
          "civilian includes airline, cargo airline, general aviation, private jet, cropduster, firefighting, and civil research. "
          "If the operator is unclear, make the best evidence-based visual judgment. "
          "Answer strictly as JSON, e.g. {\"label\":\"MILITARY\",\"reason\":\"very short visual justification\"}. "
          "Use only one of MILITARY or CIVILIAN. No additional output.")
LABELS=['civilian','military']
LABELS_RU = {'civilian': 'Гражданский', 'military': 'Военный'}
DIFFICULTY_RU = {'easy': 'Простой', 'hard': 'Сложный'}
STATUS_RU = {'ok': 'Успешно', 'api_error': 'Ошибка API', 'parse_error': 'Ошибка ответа'}


def load_manifest(path=MANIFEST):
    df=pd.read_csv(path, dtype={'id':str},encoding='utf-8-sig').fillna('')
    assert len(df)==30, 'The selected set must contain 30 unique airplanes'
    assert df['id'].is_unique, 'Duplicate IDs'
    assert df['label'].value_counts().to_dict()=={'civilian':15,'military':15}, 'Class imbalance'
    assert set(df['difficulty'])=={'hard','easy'}
    return df


def _hashpath(filename):
    # Commons uses an md5 tree of the UTF-8 filename, with spaces replaced by underscores.
    normalized=filename.replace(' ','_')
    h=hashlib.md5(normalized.encode('utf-8')).hexdigest()
    encoded=quote(normalized,safe="()!'$,-._")
    return f'https://upload.wikimedia.org/wikipedia/commons/{h[0]}/{h[:2]}/{encoded}', f'https://upload.wikimedia.org/wikipedia/commons/thumb/{h[0]}/{h[:2]}/{encoded}/640px-{encoded}'


def get_commons_info(filename, timeout=25):
    response=SESSION.get(API, params={'action':'query','format':'json','prop':'imageinfo',
                'iiprop':'url|extmetadata','iiurlwidth':640,'titles':'File:'+filename},timeout=timeout)
    response.raise_for_status()
    pages=response.json()['query']['pages']
    data=list(pages.values())[0]
    if 'missing' in data or 'imageinfo' not in data:
        raise ValueError(f'Commons file unavailable: {filename}')
    info=data['imageinfo'][0]
    md=info.get('extmetadata',{})
    def v(key):return unescape(re.sub(r'<[^>]+>','',md.get(key,{}).get('value',''))).strip()
    return dict(image_url=info.get('thumburl',info.get('url')), full_url=info.get('url',''),
        author=v('Artist'),license=v('LicenseShortName'),license_url=v('LicenseUrl'),credit=v('Credit'))


def _download_first_valid(urls, dst, timeout=35):
    errors=[]
    for url in dict.fromkeys(u for u in urls if u):
        try:
            rsp=SESSION.get(url,timeout=timeout)
            rsp.raise_for_status()
            with Image.open(io.BytesIO(rsp.content)) as img:
                img.verify()
            with Image.open(io.BytesIO(rsp.content)) as img:
                if img.width < 160 or img.height < 120:
                    raise ValueError(f'Image too small: {img.size}')
                rgb=ImageOps.exif_transpose(img).convert('RGB')
                rgb.thumbnail((960,960),Image.Resampling.LANCZOS)
                temp=dst.with_suffix('.tmp')
                rgb.save(temp,format='JPEG',quality=87,optimize=True)
                temp.replace(dst)
            return url
        except Exception as exc:
            errors.append(f'{url[:95]}: {type(exc).__name__}: {str(exc)[:120]}')
    raise RuntimeError('All Commons download methods failed. '+ ' | '.join(errors))


def download_dataset(df=None, directory=IMAGES, max_items=30, wait=0.25):
    """Download selected freely accessible Commons photos. Skips valid cached images.

    Metadata lookup is optional so simple redirects can work even when the API is blocked.
    Failed downloads are recorded, not replaced with synthetic images.
    """
    if df is None:df=load_manifest()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    failures=[]; records=[]
    for _,item in df.head(max_items).iterrows():
        path=directory/f"{item['id']}.jpg"
        info={}
        source_used='cached'
        if path.exists():
            try:
                with Image.open(path) as im: im.verify()
            except (UnidentifiedImageError,OSError):path.unlink()
        if not path.exists():
            try:
                info=get_commons_info(item['commons_file'])
            except Exception as exc:
                info={'metadata_error':f'{type(exc).__name__}: {str(exc)[:180]}'}
            orig,thumb=_hashpath(item['commons_file'])
            special='https://commons.wikimedia.org/wiki/Special:Redirect/file/'+quote(item['commons_file'].replace(' ','_'),safe="()!'$,-._")+'?width=640'
            try:
                source_used=_download_first_valid([info.get('image_url'),thumb,special,info.get('full_url'),orig],path)
            except Exception as exc:
                failures.append({'id':item['id'],'file':item['commons_file'],'error':str(exc)[:900]})
                continue
            time.sleep(wait)
        records.append({'id':item['id'],'aircraft':item['aircraft'], 'source_page':item['source_page'],
            'author':info.get('author','смотрите страницу Commons'), 'license':info.get('license','см. страницу Commons'),
            'license_url':info.get('license_url',''), 'download_source':source_used,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    pd.DataFrame(records).to_csv(OUT/'credits.csv',index=False,encoding='utf-8-sig')
    if failures:
        pd.DataFrame(failures,columns=['id','file','error']).to_csv(OUT/'download_errors.csv',index=False,encoding='utf-8-sig')
    elif (OUT/'download_errors.csv').exists():
        (OUT/'download_errors.csv').unlink()
    print(f'Загружено фотографий: {len(records)} из {min(max_items,len(df))}. Ошибок: {len(failures)}.')
    return records,failures


def encode_image(path, max_dim=MAX_IMAGE_DIM):
    """Encode resized pixels, NOT the filename/ground-truth, for Ollama's REST API."""
    with Image.open(path) as img:
        im=ImageOps.exif_transpose(img).convert('RGB')
        im.thumbnail((max_dim,max_dim),Image.Resampling.LANCZOS)
        out=io.BytesIO()
        im.save(out,format='JPEG',quality=82,optimize=True)
    return base64.b64encode(out.getvalue()).decode('ascii')


def parse_label(answer):
    """Conservative parsing: no ambiguous natural-language guesses."""
    if not isinstance(answer,str) or not answer.strip():
        return None
    text=answer.strip()
    block=re.search(r'\{[^{}]+\}',text,re.DOTALL)
    if block:
        try:
            obj=json.loads(block.group())
            val=str(obj.get('label','')).strip().lower()
            if val in LABELS:return val
            if val in ('военный','военное','военная'):return 'military'
            if val in ('гражданский','гражданское','гражданская'):return 'civilian'
        except (ValueError,TypeError,AttributeError):pass
    # Recover an unambiguous leading JSON label even if the explanation is truncated.
    # This is not a claim that the entire JSON is valid.
    recovered=re.match(r'^\s*\{\s*"label"\s*:\s*"(MILITARY|CIVILIAN)"\s*(?:,|\})',text,re.I)
    if recovered and len(re.findall(r'"label"\s*:',text,re.I))==1:
        return recovered.group(1).lower()
    value=re.sub(r'^[\s`"\']+|[\s`"\'.!?]+$','',text).lower()
    if value in LABELS:return value
    if value in ('военный','военное','военная'):return 'military'
    if value in ('гражданский','гражданское','гражданская'):return 'civilian'
    return None


def is_valid_json_response(answer):
    """Strict format check, separate from recoverable classification label."""
    try:
        obj=json.loads(answer)
        return isinstance(obj,dict) and str(obj.get('label','')).strip().lower() in LABELS
    except (ValueError,TypeError,AttributeError):
        return False


def get_ollama_models(host=OLLAMA_HOST,timeout=5):
    """Return model->digest mapping; raise helpful exception if server not running."""
    try:
        rsp=SESSION.get(host+'/api/tags', timeout=timeout)
        rsp.raise_for_status()
        return {row['name']:row.get('digest','') for row in rsp.json().get('models',[])}
    except requests.RequestException as exc:
        raise ConnectionError(f'Не получается подключиться к Ollama: {host}. '
            'Установите и откройте Ollama (или выполните ollama serve). '
            f'Подробности: {exc}') from exc


def check_models(model_ids=MODELS, host=OLLAMA_HOST):
    installed=get_ollama_models(host)
    missing=[name for name in model_ids if name not in installed]
    if missing:
        raise RuntimeError('Модели не загружены: '+', '.join(missing)+'. В терминале: '
             + ' ; '.join('ollama pull '+x for x in missing))
    return {name:installed[name] for name in model_ids}


def build_signature(model,digest,image_hash,opts=OPTIONS,max_dim=MAX_IMAGE_DIM):
    obj={'model':model, 'model_digest':digest, 'image_sha256':image_hash,
         'prompt':PROMPT,'options':opts,'max_image_dim':max_dim,'api':'ollama-chat-v1'}
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()


def predict_one(model_id, path, host=OLLAMA_HOST, options=OPTIONS, session=None,
                timeout_s=300, tries=2, max_dim=MAX_IMAGE_DIM):
    """Exactly one image + the identical fixed prompt to POST /api/chat."""
    session=session or SESSION
    payload={'model':model_id, 'stream':False, 'format':'json',
             'messages':[{'role':'user','content':PROMPT,
                          'images':[encode_image(path,max_dim=max_dim)]}],
             'options':dict(options), 'keep_alive':'5m'}
    started=time.perf_counter()
    last_error=None
    for attempt in range(tries):
        try:
            response=session.post(host+'/api/chat', json=payload,timeout=timeout_s)
            response.raise_for_status()
            data=response.json()
            raw=str(data.get('message',{}).get('content',''))
            label=parse_label(raw)
            return {'predicted':label or '', 'raw_response':raw,
                    'status':'ok' if label else 'parse_error','json_valid':is_valid_json_response(raw),'error':'',
                    'latency_s':round(time.perf_counter()-started,3),
                    'eval_count':data.get('eval_count',''),
                    'total_duration_s':round(data.get('total_duration',0)/1e9,3)}
        except (requests.RequestException,ValueError,KeyError,TypeError) as exc:
            last_error=exc
            if attempt+1<tries:
                time.sleep(1.5)
    return {'predicted':'','raw_response':'','status':'api_error','json_valid':False,
            'error':f'{type(last_error).__name__}: {str(last_error)[:350]}',
            'latency_s':round(time.perf_counter()-started,3),
            'eval_count':'','total_duration_s':''}


def unload_model(model_id,host=OLLAMA_HOST,session=None):
    """Unload from RAM between models; do not unload after every picture."""
    session=session or SESSION
    response=session.post(host+'/api/generate',
             json={'model':model_id,'keep_alive':0, 'stream':False},timeout=120)
    response.raise_for_status()
    return response.json()


def run_inference(df,model_ids=MODELS,limit=30,output=OUT/'predictions.csv',
                  force=False,host=OLLAMA_HOST,options=OPTIONS,max_dim=MAX_IMAGE_DIM,
                  timeout_s=300,session=None):
    """Sequential by MODEL (not by image): prevent both large models staying resident.

    Resume with a cache key including image bytes, actual model digest, prompt & options.
    An interrupted run writes a CSV immediately after each HTTP response.
    """
    limit=int(limit)
    if not 1<=limit<=len(df):raise ValueError('MAX_IMAGES должен быть в диапазоне 1–30')
    if len(model_ids)!=2 or len(set(model_ids))!=2:
        raise ValueError('Для сравнения выберите две разные модели.')
    selected=df.iloc[:limit].copy()
    missing=[str(row.id) for row in selected.itertuples()
             if not (IMAGES/f'{row.id}.jpg').exists()]
    if missing:
        raise FileNotFoundError('Нет изображений: '+', '.join(missing)+'. Сначала выполните download_dataset.')
    digests=check_models(model_ids,host)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    columns=['id','model_name','model_id','model_digest','image_sha256','signature',
       'truth','difficulty','aircraft','predicted','status','json_valid','raw_response','error',
       'latency_s','eval_count','total_duration_s','timestamp_utc']
    old=pd.read_csv(output,dtype={'id':str}).fillna('') if output.exists() else pd.DataFrame(columns=columns)
    # Legacy result files without signature are ignored; avoid counting incompatible experiments.
    if not set(columns).issubset(set(old.columns)):
        if len(old):
            raise ValueError(f'Файл {output} имеет другую схему. Сохраните его отдельно и начните новый эксперимент.')
        old=pd.DataFrame(columns=columns)
    session=session or SESSION
    planned=limit*len(model_ids)
    complete=0
    cached_count=0
    for model_id in model_ids:
        print(f'\n=== {model_id} ({digests[model_id][:12]}) ===')
        try:
            for row in selected.itertuples(index=False):
                pic=IMAGES/f'{row.id}.jpg'
                image_hash=hashlib.sha256(pic.read_bytes()).hexdigest()
                signature=build_signature(model_id,digests[model_id],image_hash,options,max_dim)
                seen=old[(old['id']==str(row.id)) & (old['model_name']==model_id) &
                    (old['signature']==signature) & (old['status']=='ok')]
                if len(seen) and not force:
                    complete+=1
                    cached_count+=1
                    continue
                outcome=predict_one(model_id,pic,host,options,session=session,
                      timeout_s=timeout_s,max_dim=max_dim)
                rec={'id':str(row.id),'model_name':model_id,'model_id':model_id,
                     'model_digest':digests[model_id], 'image_sha256':image_hash,
                     'signature':signature,'truth':row.label,'difficulty':row.difficulty,
                     'aircraft':row.aircraft,**outcome,
                     'timestamp_utc':datetime.now(timezone.utc).isoformat()}
                # Keep at most one prediction for image+model; saved atomically on each result.
                old=old.loc[~((old['id']==str(row.id))&(old['model_name']==model_id))].copy()
                old=pd.concat([old,pd.DataFrame([rec])],ignore_index=True)
                old=old.sort_values(['model_name','id']).reset_index(drop=True)
                temp=output.with_suffix('.partial')
                old.to_csv(temp,index=False,encoding='utf-8-sig')
                temp.replace(output)
                complete+=1
                print(f'{row.id}: {LABELS_RU.get(outcome["predicted"], STATUS_RU.get(outcome["status"], outcome["status"]))} ({outcome["latency_s"]:.1f} с)')
        finally:
            try:
                unload_model(model_id,host,session=session)
            except (requests.RequestException,ValueError) as exc:
                print(f'Предупреждение: не удалось выгрузить {model_id}: {exc}')
    print(f'Обработано {complete} запросов из {planned}, из кэша: {cached_count}.')
    return pd.read_csv(output,dtype={'id':str}).fillna('')


def score_predictions(df,selected,model_names=None):
    """Main accuracy is valid only for 30/30 answered; partial scores are explicitly labeled."""
    ids=set(selected['id'].astype(str))
    data=df.copy(); data['id']=data.id.astype(str)
    data=data[data.id.isin(ids)]
    if model_names is None:model_names=sorted(data['model_name'].unique())
    rows=[]
    for name in model_names:
        preds=data[data.model_name==name].drop_duplicates('id',keep='last')
        full=selected[['id','label','difficulty']].copy();full['id']=full.id.astype(str)
        full=full.merge(preds[['id','predicted','status','latency_s']],on='id',how='left')
        full['valid']=full['predicted'].isin(LABELS) & (full['status']=='ok')
        used=full[full.valid].copy()
        count=len(used); all_count=len(full)
        if count:
            acc=accuracy_score(used['label'],used['predicted'])
            prec,rec,f1,_=precision_recall_fscore_support(used['label'],used['predicted'],labels=LABELS,average='macro',zero_division=0)
            bacc=balanced_accuracy_score(used['label'],used['predicted']) if used.label.nunique()==2 else np.nan
            cm=confusion_matrix(used['label'],used['predicted'],labels=LABELS)
            hard=used[used.difficulty=='hard'];easy=used[used.difficulty=='easy']
            hard_acc=accuracy_score(hard.label,hard.predicted) if len(hard) else np.nan
            easy_acc=accuracy_score(easy.label,easy.predicted) if len(easy) else np.nan
        else:acc=prec=rec=f1=bacc=hard_acc=easy_acc=np.nan;cm=np.zeros((2,2),dtype=int)
        lat=pd.to_numeric(used['latency_s'],errors='coerce').dropna()
        rows.append({'model':name,'n_total':all_count,'n_valid':count,
              'coverage':count/all_count if all_count else np.nan,
              'accuracy_all_30':acc if count==all_count and all_count==30 else np.nan,
              'accuracy_on_valid':acc,'balanced_accuracy_valid':bacc,
              'precision_macro_valid':prec,'recall_macro_valid':rec,'f1_macro_valid':f1,
              'accuracy_hard_valid':hard_acc,'accuracy_easy_valid':easy_acc,
              'TN':int(cm[0,0]),'FP':int(cm[0,1]),'FN':int(cm[1,0]),'TP':int(cm[1,1]),
              'median_latency_s_valid':lat.median() if len(lat) else np.nan,
              'p95_latency_s_valid':lat.quantile(.95) if len(lat) else np.nan})
    return pd.DataFrame(rows)


def paired_comparison(predictions,selected,a,b,bootstrap_repeats=3000,seed=42):
    """Paired metrics on images answered by both. McNemar exact (binomial), bootstrap 95% CI."""
    table=selected[['id','label','difficulty']].copy();table['id']=table['id'].astype(str)
    p=predictions.copy();p['id']=p.id.astype(str)
    for name,suffix in ((a,'a'),(b,'b')):
        subset=p[p.model_name==name].drop_duplicates('id',keep='last')[['id','predicted','status']]
        subset=subset.rename(columns={'predicted':f'pred_{suffix}','status':f'status_{suffix}'})
        table=table.merge(subset,on='id',how='left')
    mask=(table['pred_a'].isin(LABELS)&table['pred_b'].isin(LABELS)&
          (table['status_a']=='ok')&(table['status_b']=='ok'))
    paired=table[mask].copy()
    if paired.empty:return {'n_paired':0,'delta_accuracy_a_minus_b':np.nan,'95ci_low':np.nan,
                             '95ci_high':np.nan,'discordant_a_only':0,'discordant_b_only':0,'mcnemar_exact_p':np.nan},paired
    ia=(paired.label==paired.pred_a).to_numpy().astype(int)
    ib=(paired.label==paired.pred_b).to_numpy().astype(int)
    diffs=ia-ib
    only_a=int(((ia==1)&(ib==0)).sum());only_b=int(((ia==0)&(ib==1)).sum())
    discord=only_a+only_b
    # Two-sided exact binomial test of discordant pairs under H0: p=0.5.
    pval=(min(1.0,2*sum(math.comb(discord,k) for k in range(min(only_a,only_b)+1))/2**discord) if discord else 1.0)
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(diffs),size=(bootstrap_repeats,len(diffs)))
    bs=diffs[indices].mean(axis=1)
    result={'n_paired':len(paired),'delta_accuracy_a_minus_b':float(diffs.mean()),
        '95ci_low':float(np.quantile(bs,.025)),'95ci_high':float(np.quantile(bs,.975)),
        'discordant_a_only':only_a,'discordant_b_only':only_b,'mcnemar_exact_p':pval}
    return result,paired


def save_metrics(predictions,manifest,output_dir=OUT,model_ids=MODELS):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    names=list(model_ids)
    score=score_predictions(predictions,manifest,names)
    score.to_csv(output_dir/'metrics.csv',index=False,encoding='utf-8-sig')
    result, paired=paired_comparison(predictions,manifest,*names)
    return score,result,paired




def format_summary(predictions, manifest, model_ids=MODELS, output=OUT/'classification_vs_format.csv'):
    """Compare recovered class labels and strict JSON compliance, independently."""
    ids=set(manifest['id'].astype(str))
    data=predictions.copy()
    data['id']=data['id'].astype(str)
    rows=[]
    for model in model_ids:
        part=data[(data.model_name==model)&data.id.isin(ids)].drop_duplicates('id',keep='last')
        total=len(ids)
        extracted=part[(part.status=='ok') & part.predicted.isin(LABELS)]
        correct=int((extracted.predicted==extracted.truth).sum())
        # Older cached results may not contain json_valid; check original response directly.
        valid_json=int(part.raw_response.fillna('').map(is_valid_json_response).sum())
        rows.append({'model':model,'n_total':total,'n_labels':len(extracted),
                     'correct_labels':correct,
                     'accuracy_extracted':correct/len(extracted) if len(extracted) else float('nan'),
                     'json_valid_count':valid_json,'json_valid_share':valid_json/total})
    result=pd.DataFrame(rows)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False,encoding='utf-8-sig')
    return result
