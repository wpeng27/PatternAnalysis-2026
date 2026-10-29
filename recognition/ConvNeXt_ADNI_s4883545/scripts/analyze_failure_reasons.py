"""Validation-only follow-up for texture, resampling and slice-context hypotheses.

Diagnostic variants are fixed before inference. They do not replace the saved
preprocessing, checkpoint, decision boundary or review policy.
AI-assisted implementation.
"""

# Support direct script execution from any working directory.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paths import source_path

import argparse
from datetime import datetime, timezone
import csv
import io
import json
import time

import numpy as np
from PIL import Image
from train import seed_everything, PREPROCESSING_BY_SCALING
from dataset import ADNIDataset, load_splits
from modules import build_model, checkpoint_weight_kind
from scripts.compare_validation import verify_run, file_hash
from scripts.benchmark_latency import state_hash
from scripts.analyze_sensitivity import compare_probabilities, infer
import torch
import torch.nn.functional as F


VARIANTS = ('identity', 'blur_0.5', 'blur_1.0', 'roundtrip_216', 'source_bicubic', 'aspect_preserving')
DEFINITIONS = {
    'identity': 'Existing Pillow L -> bilinear 224x224 -> float32 [0,1]',
    'blur_0.5': 'After existing resize, 5x5 normalized Gaussian convolution, sigma 0.5 pixel, reflect padding',
    'blur_1.0': 'After existing resize, 5x5 normalized Gaussian convolution, sigma 1.0 pixel, reflect padding',
    'roundtrip_216': 'Resize full 224x224 image to 216x216 then back to 224x224, no crop, torch bilinear align_corners=False antialias=True',
    'source_bicubic': 'Replace only original source JPEG resize with Pillow bicubic 224x224',
    'aspect_preserving': 'Fit source grayscale JPEG within 224x224 using bilinear resize, rounded dimensions, symmetric zero padding with odd extra pixel at bottom/right. Changes framing and resampling as well as aspect ratio.',
    'neighbour_mean': 'Arithmetic mean P(AD) of this slice and the closest available lower/higher slice IDs within the same patient, two slices at endpoints, three elsewhere. No physical slice spacing assumed.',
    'other_slices_mean': 'Mean P(AD) of every other available slice of the same patient, excluding the target slice',
    'limits': 'Exploratory tests of model sensitivity, not anatomical diagnosis or proof of clinically label-preserving transforms. Context predictions require additional images and are secondary to individual-slice evaluation.',
}


def tensor_variant(x, name):
    if x.ndim != 4 or x.shape[1] != 1 or not x.is_floating_point():
        raise ValueError('Expected floating N,1,H,W tensor')
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('Expected finite unit-range pixels')
    if name == 'identity':
        return x.clone()
    if name == 'roundtrip_216':
        if x.shape[-2:] != (224, 224):
            raise ValueError('Roundtrip requires 224x224 input')
        small = F.interpolate(x, size=(216,216), mode='bilinear', align_corners=False, antialias=True)
        return F.interpolate(small, size=(224,224), mode='bilinear', align_corners=False, antialias=True)
    if name in ('blur_0.5', 'blur_1.0'):
        sigma = float(name.split('_')[1])
        t = torch.arange(-2,3,dtype=x.dtype,device=x.device)
        kernel = torch.exp(-t.square()/(2*sigma*sigma))
        kernel = kernel/kernel.sum()
        weight = torch.outer(kernel,kernel).reshape(1,1,5,5)
        return F.conv2d(F.pad(x,(2,2,2,2),mode='reflect'),weight).clamp(0,1)
    raise ValueError(f'Unknown tensor variant: {name}')


def source_variant(image, name):
    gray = image.convert('L')
    if name == 'source_bicubic':
        resized = gray.resize((224,224),Image.Resampling.BICUBIC)
    elif name == 'aspect_preserving':
        width,height = gray.size
        factor = min(224/width,224/height)
        shape = (max(1,round(width*factor)),max(1,round(height*factor)))
        resized = Image.new('L',(224,224),0)
        resized.paste(gray.resize(shape,Image.Resampling.BILINEAR),((224-shape[0])//2,(224-shape[1])//2))
    else:
        raise ValueError(f'Unknown source variant: {name}')
    return torch.from_numpy(np.array(resized,copy=True)).float().div(255).unsqueeze(0)


def context_predictions(rows, probabilities):
    if len(rows)!=len(probabilities) or len({r['path'] for r in rows})!=len(rows):
        raise ValueError('Need unique aligned rows')
    groups={}
    for i,row in enumerate(rows):
        if not np.isfinite(probabilities[i]) or not 0<=probabilities[i]<=1:
            raise ValueError('Invalid probability')
        groups.setdefault(row['patient_id'],[]).append(i)
    neighbour,other=np.empty(len(rows)),np.empty(len(rows))
    for indices in groups.values():
        if len(indices)<2 or len({rows[i]['label'] for i in indices})!=1:
            raise ValueError('Need at least two consistently labelled slices per patient')
        ordered=sorted(indices,key=lambda i:int(rows[i]['slice_id']))
        if len({int(rows[i]['slice_id']) for i in indices})!=len(indices):
            raise ValueError('Repeated slice identifier')
        for position,i in enumerate(ordered):
            adjacent=ordered[max(0,position-1):position+2]
            neighbour[i]=sum(probabilities[j] for j in adjacent)/len(adjacent)
            other[i]=sum(probabilities[j] for j in ordered if j!=i)/(len(ordered)-1)
    return {'neighbour_mean':neighbour.tolist(),'other_slices_mean':other.tolist()}


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not torch.cuda.is_available():
        parser.error('Need a fresh output directory and a CUDA GPU')
    torch.set_num_threads(4)
    seed_everything(42)
    torch.backends.cuda.matmul.fp32_precision='ieee'
    torch.backends.cudnn.fp32_precision='ieee'
    analysis=json.loads(args.analysis.read_text(encoding='utf-8'))
    cases=json.loads(args.cases.read_text(encoding='utf-8'))['cases']
    splits,audit=load_splits(Path(analysis['inputs']['splits']))
    rows=splits['val']
    lookup={r['path']:i for i,r in enumerate(rows)}
    indices=[lookup[c['path']] for c in cases]
    labels=[r['label'] for r in rows]
    verified,original={},{}
    for entry in analysis['models']:
        name=entry['model']
        run=Path(entry['verification']['run'])
        v,p=verify_run(run,run.with_name(run.name+'_validation'),name,rows,audit)
        for k in ('checkpoint_sha256','prediction_sha256'):
            assert v[k]==entry['verification'][k]
        verified[name],original[name]=v,p
    for c,i in zip(cases,indices):
        assert c['p_ad']==original['convnext_tiny'][i] and c['patient_id']==rows[i]['patient_id']
    thresholds={m['model']:m['selected']['threshold'] for m in analysis['models']}
    args.output.mkdir(parents=True)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'variants':VARIANTS,'definitions':DEFINITIONS,
          'rows':len(rows),'patients':len({r['patient_id'] for r in rows}),'case_paths':[c['path'] for c in cases],
          'test_evaluated':False,'training_performed':False,'thresholds':thresholds,'split_sha256':audit['split_sha256'],
          'analysis_sha256':file_hash(args.analysis),'case_evidence_sha256':file_hash(args.cases),
          'source_sha256':{name:file_hash(source_path(name)) for name in
                           ('analyze_failure_reasons.py','analyze_sensitivity.py','dataset.py','modules.py','metrics.py','compare_validation.py')},
          'models':{k:{f:v[f] for f in ('run','checkpoint_sha256','prediction_sha256','selected_epoch')} for k,v in verified.items()},
          'seed':42,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__)}
    write_json(args.output/'plan.json',plan)
    write_json(args.output/'status.json',{'state':'running'})
    def log(message):
        print(message,flush=True)
        with (args.output/'run.log').open('a',encoding='utf-8') as f:
            f.write(message+'\n')
    log('Plan saved. Loading audited validation images.')
    start=time.perf_counter()
    root=Path(analysis['inputs']['data_root'])
    dataset=ADNIDataset(root,rows,'unit','none')
    pool=torch.empty(len(rows),1,224,224)
    for i in range(len(rows)):
        image,label=dataset[i]
        assert label==labels[i] and torch.equal(image[0],image[1]) and torch.equal(image[1],image[2])
        pool[i]=image[:1]
    models={}
    result={'plan':plan,'models':{}}
    for key,v in verified.items():
        checkpoint=torch.load(Path(v['run'])/'best.pt',map_location='cpu',weights_only=True)
        kind=checkpoint_weight_kind(checkpoint)
        config=checkpoint['config']
        scaling=config.get('input_scaling','unit')
        assert config['preprocessing']==PREPROCESSING_BY_SCALING[scaling]
        model=build_model(key,**config.get('model_kwargs',{}))
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        model.eval().requires_grad_(False).cuda()
        models[key]={'model':model,'state_hash':state_hash(model),'scaling':scaling,
                     'batch':v['settings']['evaluation_batch_size'],
                     'dtype':config.get('amp_dtype','float16') if config['effective_amp'] else 'float32'}
        result['models'][key]={'weights':kind,'batch':models[key]['batch'],'precision':models[key]['dtype'],
                              'variants':{},'context':{}}
        del checkpoint
    with (args.output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['model','condition','path','patient_id','label','p_ad'])
        for variant in VARIANTS:
            tick=time.perf_counter()
            transformed=torch.empty_like(pool)
            if variant in ('source_bicubic','aspect_preserving'):
                for i,row in enumerate(rows):
                    path=root/row['path']
                    raw=path.read_bytes()
                    import hashlib
                    assert hashlib.sha256(raw).hexdigest()==row['file_sha256']
                    with Image.open(io.BytesIO(raw)) as image:
                        transformed[i]=source_variant(image,variant)
            else:
                for offset in range(0,len(rows),32):
                    transformed[offset:offset+32]=tensor_variant(pool[offset:offset+32],variant)
            for key,m in models.items():
                probabilities=infer(m['model'],transformed,'identity',m['scaling'],m['batch'],m['dtype'])
                if variant=='identity':
                    assert probabilities==original[key], 'Unchanged inference must reproduce the saved export exactly'
                values=compare_probabilities(labels,original[key],probabilities,thresholds[key])
                values['cases']=[probabilities[i] for i in indices]
                values['case_fp32']=infer(m['model'],transformed[indices],'identity',m['scaling'],len(indices),'float32')
                result['models'][key]['variants'][variant]=values
                for row,p in zip(rows,probabilities):
                    writer.writerow([key,variant,row['path'],row['patient_id'],row['label'],p])
                f.flush()
                log(f"{key} {variant}: accuracy={values['metrics']['accuracy']:.6f}, corrected={values['wrong_to_correct']}, harmed={values['correct_to_wrong']}, cases={values['cases']}")
            result.setdefault('variant_seconds',{})[variant]=time.perf_counter()-tick
            write_json(args.output/'results.partial.json',result)
            del transformed
        for key,m in models.items():
            for name,probabilities in context_predictions(rows,original[key]).items():
                values=compare_probabilities(labels,original[key],probabilities,thresholds[key])
                values['cases']=[probabilities[i] for i in indices]
                values['interpretation']='Secondary per-slice context predictions, requiring other patient slices. Not primary individual-slice evaluation.'
                result['models'][key]['context'][name]=values
                for row,p in zip(rows,probabilities):
                    writer.writerow([key,name,row['path'],row['patient_id'],row['label'],p])
            assert state_hash(m['model'])==m['state_hash']
            assert file_hash(Path(verified[key]['run'])/'best.pt')==verified[key]['checkpoint_sha256']
            result['models'][key]['weights_unchanged']=True
    result.update(state='completed',seconds=time.perf_counter()-start,completed_utc=datetime.now(timezone.utc).isoformat())
    assert file_hash(args.analysis)==plan['analysis_sha256'] and file_hash(args.cases)==plan['case_evidence_sha256']
    write_json(args.output/'results.json',result)
    write_json(args.output/'status.json',{k:result[k] for k in ('state','seconds','completed_utc')})
    log(f"Completed in {result['seconds']:.2f} seconds. No training or test evaluation.")


if __name__=='__main__':
    main()
