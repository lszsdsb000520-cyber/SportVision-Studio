# OpenTTGames 下载脚本（120fps 全高清乒乓球长视频 + 落点/球坐标标注）
#
# 用法（PowerShell）：
#   cd D:\code\pingpongtrack
#   powershell -ExecutionPolicy Bypass -File tools\download_openttgames.ps1 -Set test
#   powershell -ExecutionPolicy Bypass -File tools\download_openttgames.ps1 -Set train -Names game_3,game_4
#   powershell -ExecutionPolicy Bypass -File tools\download_openttgames.ps1 -Set all
#
# 输出目录：
#   datasets\openttgames\videos\test\test_2.mp4
#   datasets\openttgames\ann\test\test_2\{ball_markup.json,events_markup.json,segmentation_masks\}
#
# 说明：
#   * 视频和 zip 走断点续传（curl.exe --continue-at -），中断后重跑即可。
#   * zip 只含标注，体积很小；视频很大，先下 test 段试通再下 train。
#   * 数据许可 CC BY-NC-SA 4.0（仅限非商业用途），引用见 lab.osai.ai。

param(
    [ValidateSet('test', 'train', 'all')]
    [string]$Set = 'test',
    [string[]]$Names = @(),
    [string]$OutDir = (Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) 'datasets\openttgames'),
    [switch]$SkipVideo,
    [switch]$SkipAnn
)

$ErrorActionPreference = 'Stop'
$base = 'https://lab.osai.ai/datasets/openttgames/data'

# 名称 -> 大小(MB)，仅用于进度提示和下载后校验下界
$sizes = @{
    'game_1' = 5632; 'game_2' = 10800; 'game_3' = 4610; 'game_4' = 3990; 'game_5' = 4510;
    'test_1' = 1100; 'test_2' = 225;   'test_3' = 488;  'test_4' = 2300; 'test_5' = 720;
    'test_6' = 676;  'test_7' = 488
}

$list = switch ($Set) {
    'test' { @('test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7') }
    'train' { @('game_1', 'game_2', 'game_3', 'game_4', 'game_5') }
    'all' { @('test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7', 'game_1', 'game_2', 'game_3', 'game_4', 'game_5') }
}
if ($Names.Count -gt 0) { $list = $list | Where-Object { $Names -contains $_ } }
if (-not $list) { throw "没有匹配的条目：$Names" }

$curl = (Get-Command curl.exe -ErrorAction SilentlyContinue)
$vidDir = Join-Path $OutDir 'videos'
$annDir = Join-Path $OutDir 'ann'
$split = if ($Set -eq 'train') { 'train' } else { 'test' }

function Get-File([string]$url, [string]$dest) {
    $dir = Split-Path -Parent $dest
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    if ($curl) {
        & $curl.Source -L --fail --retry 3 --continue-at - --silent --show-error -o $dest $url
        if ($LASTEXITCODE -ne 0) { throw "下载失败：$url" }
    } else {
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    }
}

$totalMB = 0
foreach ($n in $list) { $totalMB += ($sizes[$n] -as [int]) }
Write-Host ("准备下载 {0} 个条目，约 {1:N0} MB 视频 + 少量标注 -> {2}" -f $list.Count, $totalMB, $OutDir)

foreach ($n in $list) {
    $sub = if ($n -like 'test_*') { 'test' } else { 'train' }
    if (-not $SkipVideo) {
        $mp4 = Join-Path (Join-Path $vidDir $sub) ($n + '.mp4')
        if (Test-Path $mp4) {
            Write-Host ("[跳过] 已存在 {0} ({1:N0} MB)" -f $mp4, ((Get-Item $mp4).Length / 1MB))
        } else {
            Write-Host ("[视频] {0}.mp4  ~{1:N0} MB" -f $n, $sizes[$n])
            Get-File "$base/$n.mp4" $mp4
            $mb = (Get-Item $mp4).Length / 1MB
            if ($mb -lt ($sizes[$n] * 0.5)) { Write-Warning ("{0} 只有 {1:N0} MB，可能没下完，重跑本脚本续传" -f $n, $mb) }
        }
    }
    if (-not $SkipAnn) {
        $zip = Join-Path (Join-Path $annDir $sub) ($n + '.zip')
        $out = Join-Path (Join-Path $annDir $sub) $n
        if (Test-Path (Join-Path $out 'events_markup.json')) {
            Write-Host ("[跳过] 标注已解压 {0}" -f $out)
        } else {
            Write-Host ("[标注] {0}.zip" -f $n)
            Get-File "$base/$n.zip" $zip
            Expand-Archive -Path $zip -DestinationPath $out -Force
        }
    }
}

Write-Host ''
Write-Host '完成。目录结构：'
Get-ChildItem $OutDir -Recurse -Depth 2 -Directory | Select-Object -ExpandProperty FullName
Write-Host ''
Write-Host '标注格式：'
Write-Host '  events_markup.json : {"92":"net","109":"bounce"}   ← 帧号 -> 事件（bounce=触台落点，net=拦网）'
Write-Host '  ball_markup.json   : {"88":{"x":858,"y":497}}     ← 帧号 -> 球心像素坐标（事件前后各若干帧）'
Write-Host '  segmentation_masks\: 每帧一张 png（人/球台/记分牌通道编码）'
