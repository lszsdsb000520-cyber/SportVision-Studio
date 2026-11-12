"""Publish a reproducible summary using measured results only."""
from pathlib import Path
import argparse,json,shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    report=ROOT/'reports/extended';result=json.loads((report/'training_result.json').read_text())
    run=Path(result['run']);config=json.loads((run/'config.json').read_text())
    decision=json.loads((report/'deployment_decision.json').read_text(encoding='utf-8'))
    threshold=result['selected_threshold'];base_threshold=result['baseline_calibrated_threshold']
    rows=[('原模型默认',.5,result['baseline_validation']['0.5'],result['baseline_test']['0.5']),
          ('原模型仅校准阈值',base_threshold,result['baseline_validation'][str(base_threshold)],result['baseline_test'][str(base_threshold)]),
          ('本轮微调模型',threshold,result['selected_validation'][str(threshold)],result['selected_test'])]
    table=['| 模型 | 阈值 | 验证 F1 | 保留测试精确率 | 保留测试召回率 | 保留测试 F1 |',
           '|---|---:|---:|---:|---:|---:|']
    for title,t,val,test in rows:
        table.append(f"| {title} | {t:.2f} | {val['f1']:.2%} | {test['precision']:.2%} | {test['recall']:.2%} | {test['f1']:.2%} |")
    lines=['# 新数据训练与评估报告','',
        '## 做了什么','',
        '使用新增 ExtendedOpenTTGames 标注，并从官方视频按需抽取对应帧。明确有球坐标或明确无球的帧才参与监督，缺失标签不视作无球。保持原始导入数据不变。','',
        '训练：game_1、game_2、game_3、game_5；验证：game_4；保留测试：官方 test_3、test_5、test_7。按完整来源分组，所取样本在集合间没有完全相同图像。',
        f"样本数：训练 {result['train_windows']*3} 帧、验证 {result['val_windows']*3} 帧、测试 {result['test_windows']*3} 帧。",'',
        f"训练 {config['epochs']} 轮，随机种子 {config['seed']}；BlurBall 只微调 stage4 和输出头，其余参数及批归一化统计冻结；AdamW，学习率 {config['lr']}，batch {config['batch']}，AMP。验证集选择第 {result['best_epoch']} 轮，阈值 {threshold:.2f}，选择完成后才评估测试样本。",'',
        '## 验证结果','',*table,'',
        'F1是小球定位指标：在512×288坐标系中，与标注距离不超过6像素记为命中；错位同时计一次误检和一次漏检。该数值不是落点、出界或拦网识别准确率。','',
        f"新测试的漏检计数：{result['baseline_test']['0.5']['fn']} → {result['selected_test']['fn']}；误检计数：{result['baseline_test']['0.5']['fp']} → {result['selected_test']['fp']}。",
        f"旧场景回归（566帧）：F1 {decision['original_scene_baseline_f1']:.2%} → {decision['original_scene_candidate_f1']:.2%}，因此保留原通用默认模型。",'',
        f"实验耗时 {result['elapsed_seconds']/60:.1f} 分钟；PyTorch峰值分配显存 {result['peak_vram_bytes']/1024**2:.0f} MiB；设备 {config['gpu']}，PyTorch {config['torch']}，CUDA {config['cuda']}。",'',
        f"部署处理：{decision['summary']}",'',
        '导出使用固定1×9×288×512输入的ONNX，并核对真实输入上的数值误差、运行后端和任务指标。旧场景回归及部署检查见随附JSON；训练曲线为 curve.png。','',
        '## 遗留问题','',
        '- 这是固定抽样子集，不是全量数据集或比赛官方分数；原球坐标带有半自动标注性质，公开预训练权重的原训练成员也无法独立审计。',
        '- 新数据的普通 net 标签包含可以继续回合的碰网；回合结束标签又包含本方台面失误。两者不能全部作为“被网拦停”标签混训。本轮以球检测微调为主，事件窗口诊断单独提供，不将其写成整场落点准确率。',
        '- 自动分析结果仍需人工复核；合格视频继续按“导入名_success.原扩展名”入库，未核对的球心不自动成为训练真值。',
        '- 数据集采用 CC BY-NC-SA 4.0；应保留来源、署名和授权说明。','',
        '来源：[ExtendedOpenTTGames](https://github.com/moamal01/table_tennis_data)、[OpenTTGames官方数据](https://lab.osai.ai/)。','']
    print('\n'.join(lines))
    for name in ('config.json','metrics.json','result.json','selection.json','curve.png'):
        shutil.copy2(run/name,out/name)
    for name in ('annotation_audit.json','data_verification.json','deployment_decision.json','runtime_validation.json',
                 'regression_baseline.json','regression_candidate.json','bounce_diagnostic.json'):
        if (report/name).exists():shutil.copy2(report/name,out/name)
    shutil.copy2(ROOT/'datasets/extended/manifest.json',out/'data_manifest.json')


if __name__=='__main__':main()
