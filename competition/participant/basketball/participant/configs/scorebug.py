"""Find a persistent broadcast scoreboard from the clip itself.

Absence is evidence for replay only when the clip contains a credible overlay
template. Feeds without a stable scoreboard fall back to the scene classifier.
"""
import numpy as np


def on_air(samples):
    if len(samples)<15:
        return None
    ids = np.array([item[0] for item in samples])
    probs = np.array([item[1] for item in samples])
    gray = np.asarray([item[2] for item in samples],np.float32)
    _,h,w = gray.shape
    patch = (4,10)
    ph,pw = patch
    positions = [(y,x) for y in range(int(h*.77),int(h*.96)-ph+1,2)
                 for x in range(int(w*.05),int(w*.95)-pw+1,4)]
    patches = np.asarray([gray[:,y:y+ph,x:x+pw] for y,x in positions])
    best = None
    for start in range(0,len(samples)-9,5):
        segment = gray[start:start+10]
        if np.quantile(probs[start:start+10],.2)<.74:
            continue
        change = np.abs(np.diff(segment,axis=0)).mean(axis=(1,2))
        if np.median(change)<4.5 or change.max()>35:
            continue
        x = patches[:,start:start+10]
        ref = np.median(x,axis=1)
        variability = np.abs(x-ref[:,None]).mean(axis=(1,2,3))
        texture = ref.std(axis=(1,2))
        quality = texture/(3+variability)
        # Choose nearby textured pieces, avoiding a static boundary/black band.
        ok = (variability<5)&(texture>25)&(ref.mean(axis=(1,2))>35)
        for j in np.argsort(np.where(ok,quality,-1))[-8:]:
            if not ok[j]:
                continue
            y0,x0 = positions[j]
            near = [k for k,(y,x) in enumerate(positions) if ok[k]
                    and abs(y-y0)<=5 and 8<=abs(x-x0)<=34]
            if len(near)<2:
                continue
            ranked = [j]
            for k in sorted(near,key=lambda k:-quality[k]):
                if all(abs(positions[k][1]-positions[q][1])>=8 for q in ranked):
                    ranked.append(k)
            if len(ranked)<3:
                continue
            ranked = ranked[:5]
            err = np.abs(patches[ranked]-ref[ranked,None]).mean(axis=(2,3))
            hits = (err<=np.maximum(10,2.8*variability[ranked])[:,None]).sum(axis=0)
            present = hits >= max(2,(len(ranked)+1)//2)
            coverage = np.mean(present[probs>.74])
            if coverage < .38:
                continue
            score = coverage * min(1.,float(np.mean(quality[ranked]))/9)
            if best is None or score>best[0]:
                best = score,present
    if best is None:
        return None
    valid = best[1]
    # Only suppress sustained absence. Transient logo/clock changes must not
    # erase otherwise valid frames.
    suppress = []
    start = None
    for k,ok in enumerate(np.r_[valid,True]):
        if not ok and start is None:
            start = k
        if ok and start is not None:
            # A missing clock can also be a regular graphics change.
            # Require independent evidence of a replay/close-up somewhere
            # in the same absence interval, bracketed by valid scorebugs.
            low_scene = np.count_nonzero(probs[start:k]<.30)
            if start>=3 and k<len(valid) and k-start>=5 and low_scene>=max(3,.15*(k-start)):
                suppress.append((int(ids[start]),int(ids[k-1])))
            start = None
    return suppress
