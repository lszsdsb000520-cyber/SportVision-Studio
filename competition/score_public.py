"""Local reproduction of documented public-set accuracy metrics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent


def read(path):
    with open(path, encoding="utf8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def match(cost, gate):
    if not gate.any():
        return []
    # Prioritize the number of legal matches, then their costs.
    miss = (min(gate.shape)+1)*(1+float(np.max(np.abs(cost[gate]))))
    r,c = linear_sum_assignment(np.where(gate, cost, miss))
    return [(i,j) for i,j in zip(r,c) if gate[i,j]]


def pingpang(pred, truth, manifest):
    stats = defaultdict(lambda: np.zeros(4, dtype=int))
    offsets = defaultdict(list)
    details = {}
    for m in manifest:
        g = [t for t in truth if t["video_id"] == m["video_id"] and t["valid"]]
        ignore = [t for t in truth if t["video_id"] == m["video_id"] and not t["valid"]]
        p = [t for t in pred if t["video_id"] == m["video_id"]]
        def pair(a,b):
            a,b = [np.asarray([[v[k] for k in ("frame_id","x","y")] for v in seq]).reshape(-1,3)
                   for seq in (a,b)]
            t = abs(a[:,None,0]-b[None,:,0])
            d = np.linalg.norm(a[:,None,1:]-b[None,:,1:], axis=2)
            return match(t+d/50,(t<=1)&(d<=50))
        hit = pair(p,g)
        used = {i for i,j in hit}
        skipped = pair([q for i,q in enumerate(p) if i not in used], ignore)
        row = [len(hit),len(p)-len(hit)-len(skipped),len(g)-len(hit),len(skipped)]
        stats[m["angle"]] += row
        stats["all"] += row
        errors = [(abs(p[i]["frame_id"]-g[j]["frame_id"]),
                   np.hypot(p[i]["x"]-g[j]["x"],p[i]["y"]-g[j]["y"])) for i,j in hit]
        offsets["all"].extend(errors)
        offsets[m["angle"]].extend(errors)
        details[m["video_id"]] = dict(angle=m["angle"],tp=row[0],fp=row[1],fn=row[2],
                                      ignored=row[3])
    result = {k: dict(tp=int(t),fp=int(f),fn=int(n),ignored=int(i),
                    precision=float(t/max(1,t+f)), recall=float(t/max(1,t+n)),
                    f1=float(2*t/max(1,2*t+f+n)),
                    exact_frame_hits=int(sum(t==0 for t,d in offsets[k])),
                    matched_pixel_error_median=float(np.median([d for t,d in offsets[k]]))
                    if offsets[k] else None) for k,(t,f,n,i) in stats.items()}
    result["accuracy_points"] = sum(25*result.get(a,{}).get("f1",0) for a in ("0度","45度","90度"))
    result["per_video"] = details
    return result


def basketball(pred, truth, manifest):
    total = np.zeros(7, dtype=int)
    details = {}
    def summarize(counts):
        nt,p,tp,s,idtp,e,ng = map(int,counts)
        return dict(gt_boxes=nt, predicted_boxes=p, tp=tp, fp=p-tp, fn=nt-tp, idsw=s,
                    gt_ids=ng, idtp=idtp, idfp=p-idtp, idfn=nt-idtp, empty_frame_fp=e,
                    idf1=2*idtp/max(1,p+nt), mota=1-(p+nt-2*tp+s)/max(1,nt))
    for m in manifest:
        pf,gf = defaultdict(list),defaultdict(list)
        for items, dest in ((pred,pf),(truth,gf)):
            for t in items:
                if t["video_id"] == m["video_id"]:
                    dest[t["frame_id"]].append(t)
        pids = sorted({r["track_id"] for rs in pf.values() for r in rs})
        gids = sorted({r["track_id"] for rs in gf.values() for r in rs})
        pi,gi = [{tid:i for i,tid in enumerate(ids)} for ids in (pids,gids)]
        overlaps = np.zeros((len(pids),len(gids)),dtype=int)
        previous = {}
        nt,npred,tp,idsw,empty_fp = 0,0,0,0,0
        for f in sorted(gf.keys() | pf.keys()):
            ps,gs = pf[f],gf[f]
            nt += len(gs)
            npred += len(ps)
            empty_fp += len(ps) if not gs else 0
            ps = ps[:15]
            if not ps or not gs:
                continue
            a,b = [np.asarray([v["bbox"] for v in seq]) for seq in (ps,gs)]
            wh = np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
            inter = wh.prod(axis=2)
            iou = inter / np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]
                                     +(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter,1e-9)
            for i,j in zip(*np.nonzero(iou >= .5)):
                overlaps[pi[ps[i]["track_id"]],gi[gs[j]["track_id"]]] += 1
            pairs = match(1-iou,iou>=.5)
            tp += len(pairs)
            for i,j in pairs:
                p,g = ps[i]["track_id"],gs[j]["track_id"]
                idsw += g in previous and previous[g] != p
                previous[g] = p
        rr,cc = linear_sum_assignment(-overlaps)
        row = (nt,npred,tp,idsw,overlaps[rr,cc].sum(),empty_fp,len(gids))
        total += row
        details[m["video_id"]] = summarize(row)
    result = summarize(total)
    result["accuracy_points"] = 75*(.6*result["idf1"]+.4*result["mota"])
    result["per_video"] = details
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=("pingpang","basketball"))
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--run-status",type=Path)
    args = ap.parse_args()
    folder = ROOT/"participant"/args.task/"public_data"
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf8"))["videos"]
    result = globals()[args.task](read(args.predictions),read(folder/"gt_public.jsonl"),manifest)
    if args.run_status:
        status = json.loads(args.run_status.read_text(encoding="utf8"))
        actual = {v["video_id"]:v for v in status["videos"] if v["status"]=="ok"}
        if any(actual.get(m["video_id"],{}).get("frames") != m["n_frames"] for m in manifest):
            raise SystemExit("The run did not process every manifest frame successfully")
        if status.get("validation",{}).get("n_errors",1) != 0:
            raise SystemExit("Official output validation failed")
        result["performance"] = status["performance"]
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
