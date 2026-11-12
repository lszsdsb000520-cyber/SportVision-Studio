"""Number callouts outside the court; never silently omit a visible event."""
from math import ceil


def label_slots(items,width,height,label_width=38,line_height=20,top=30,bottom=30):
    if not items:return {}
    capacity=ceil(len(items)/2);groups=[[],[]]
    for item in sorted(items,key=lambda p:(p[1],p[0])):
        side=0 if item[1]<width/2 else 1
        if len(groups[side])>=capacity:side=1-side
        groups[side].append(item)
    result={}
    for side,group in enumerate(groups):
        group.sort(key=lambda p:(p[2],p[0]))
        spacing=min(line_height,(height-top-bottom)/max(1,len(group)))
        ys=[]
        for _,_,y in group:
            y=max(top,min(height-bottom-spacing,y-spacing/2))
            ys.append(max(y,ys[-1]+spacing) if ys else y)
        for i in range(len(ys)-1,-1,-1):
            ys[i]=min(ys[i],height-bottom-spacing if i==len(ys)-1 else ys[i+1]-spacing)
        for item,y in zip(group,ys):result[item[0]]=(2 if side==0 else width-label_width-2,y,label_width,spacing)
    return result
