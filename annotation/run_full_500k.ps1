# run_full_500k.ps1 —— 首期 50 万条标注（M1.5-C3，裁定 §7.1）
#
# 【断点续标】随时 Ctrl+C 停；重跑本脚本即从断点继续，已标的不会重标、不会重复花钱。
# 【预算硬闸】每段用 --response_limit 卡死计费 response 数；续跑时已完成部分会计入配额，
#             所以重跑【不会】突破预算（这点已由 _test_resume_quota.py 离线自测覆盖）。
# 【日志保留】每次运行追加到 logs\full_500k_<时间戳>.log；API 逐次调用日志在 annotated_full\..\api_logs\。
#
# 用法:
#   cd E:\Github\ReachyMni_Project\interaction-vla\annotation
#   .\run_full_500k.ps1              # 启动 / 续跑
#   .\run_full_500k.ps1 -DryRun      # 只看配置与已完成进度，不调 API
#
param(
    [switch]$DryRun,
    [int]$MaxWorkers = 20
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"

$PY   = ".\.venv\Scripts\python.exe"
$ROOT = "E:\Github\ReachyMni_Project\action-annotation\JoyAI-VL-Interaction"
$OUT  = ".\annotated_full"

New-Item -ItemType Directory -Force -Path .\logs | Out-Null
$LOG = ".\logs\full_500k_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")

# 配额：按试产实测 $0.000348/条 → 合计 50 万条 ≈ $174（裁定上限 $200）
# 配比理由：chat=真对话(最贵重)；narration=旁白密集(每样本~14条)；event_grounding=事件播报。
$Jobs = @(
    @{ Name = "chat";            In = "$ROOT\chat";                             Quota = 200000 },
    @{ Name = "narration";       In = "$ROOT\narration\narration.json";         Quota = 150000 },
    @{ Name = "event_grounding"; In = "$ROOT\event_grounding\event_grounding.json"; Quota = 150000 }
)

$total = ($Jobs | Measure-Object -Property Quota -Sum).Sum
Write-Host "================================================================"
Write-Host " 首期标注 $total 条 response  (预估 `$$([math]::Round($total*0.000348,0)))"
Write-Host " 输出: $OUT     日志: $LOG"
Write-Host " 断点续标: 重跑本脚本即可。Ctrl+C 可随时安全中断。"
Write-Host "================================================================"

foreach ($j in $Jobs) {
    $outDir = Join-Path $OUT $j.Name
    Write-Host ""
    Write-Host ">>>>> [$($j.Name)]  配额 $($j.Quota) 条 >>>>>" -ForegroundColor Cyan

    if ($DryRun) {
        $done = Get-ChildItem -Path $outDir -Recurse -Filter "*.done.json" -ErrorAction SilentlyContinue
        if ($done) {
            foreach ($d in $done) {
                $m = Get-Content $d.FullName -Raw | ConvertFrom-Json
                Write-Host ("   已完成 {0}: {1} sample / {2} response" -f $d.Name, $m.samples, $m.responses)
            }
        } else {
            Write-Host "   (尚无已完成回执)"
        }
        continue
    }

    & $PY annotate_actions_api.py `
        --input $j.In `
        --output $outDir `
        --response_limit $j.Quota `
        --max_workers $MaxWorkers 2>&1 | Tee-Object -FilePath $LOG -Append

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$($j.Name)] 退出码 $LASTEXITCODE —— 已标部分已落盘，重跑本脚本可续。" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "================================================================"
Write-Host " 全部段落跑完。日志: $LOG"
Write-Host " 出分布/成本/抽样:  $PY _analyze_trial.py annotated_full"
Write-Host "================================================================"
