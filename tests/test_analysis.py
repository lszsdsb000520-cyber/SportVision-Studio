import csv
from dataclasses import asdict
import numpy as np
import pytest
from pingpong.analysis import TrackPoint,homography,map_point,propose_bounces,export_csv,validate_project


def test_calibration_maps_only_table_surface():
    corners=[[100,100],[500,100],[500,800],[100,800]]
    assert map_point(300,450,corners)==[.7625,1.37]
    assert map_point(600,450,corners) is None
    with pytest.raises(ValueError):homography([[0,0],[1,1],[0,1],[1,0]])


def test_bounce_candidate_and_missing_segment_are_separate():
    points=[TrackPoint(i,i/60,250.+i*3,float(160-5*abs(i-15)-.04*(i-15)**2),.9) for i in range(31)]
    events=propose_bounces(points,60)
    assert len(events)==1 and events[0]['frame']==15
    assert not events[0]['confirmed']
    assert events[0]['table_xy'] is None
    for i in range(13,18):points[i].x=points[i].y=None
    assert propose_bounces(points,60)==[]


def test_polygon_filters_off_table_candidates():
    points=[TrackPoint(i,i/60,900.,float(160-5*abs(i-15)),.9) for i in range(31)]
    assert propose_bounces(points,60,[[100,100],[500,100],[500,800],[100,800]])==[]


def test_csv_preserves_absent_coordinates_and_confirmation(tmp_path):
    project=dict(tracks=[asdict(TrackPoint(0,0,None,None,.05))],events=[dict(
        frame=0,time=0,x=20,y=30,table_xy=None,confirmed=False,source='trajectory_rule',type='bounce_candidate')])
    export_csv(tmp_path,project)
    with (tmp_path/'tracks.csv').open(encoding='utf-8-sig') as f:row=next(csv.DictReader(f))
    assert row['x']=='' and row['y']==''
    with (tmp_path/'events.csv').open(encoding='utf-8-sig') as f:event=next(csv.DictReader(f))
    assert event['table_x_m']=='' and event['confirmed']=='False'


def test_project_rejects_invalid_coordinates():
    project=dict(format_version=1,fps=60,tracks=[asdict(TrackPoint(0,0,float('nan'),30,.8))],events=[],corners=[])
    with pytest.raises(ValueError):validate_project(project)


def test_airborne_smooth_arc_is_not_a_bounce():
    points=[TrackPoint(i,i/60,250.+i*3,float(160-(i-15)**2/2),.9) for i in range(31)]
    assert propose_bounces(points,60)==[]


def test_asymmetric_v_keeps_vertex_instead_of_early_smoothed_peak():
    points=[TrackPoint(i,i/60,250.+i*3,160+(i-15)*(4 if i<=15 else -12),.9) for i in range(31)]
    event=propose_bounces(points,60)[0]
    assert event['frame']==15 and abs(event['x']-295)<.01 and abs(event['y']-160)<.01


def test_short_missing_vertex_is_estimated_but_not_claimed_as_observed():
    points=[TrackPoint(i,i/60,250.+i*3,160-5*abs(i-15),.9) for i in range(31)]
    points[15].x=points[15].y=None
    event=propose_bounces(points,60)[0]
    assert abs(event['frame']-15)<=1 and event['vertex_observed'] is False


def test_clip_cut_does_not_create_a_bounce():
    points=[TrackPoint(i,i/60,250.+i*3,160-5*abs(i-15),.9) for i in range(31)]
    assert propose_bounces(points,60,break_frames=[15])==[]
