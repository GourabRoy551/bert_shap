@echo off
setlocal

set "EXPERIMENT_DIR=%~dp0"
set "LEGACY_PYTHON=D:\Research\TRDP_Study Note\TRDP1\trdp\sst2_quantitative_analysis\.venv\Scripts\python.exe"
set "BERT_SHAP_PACKAGES=%EXPERIMENT_DIR%..\bert_shap\.packages"

set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"

if exist "%BERT_SHAP_PACKAGES%" (
    set "PYTHONPATH=%BERT_SHAP_PACKAGES%;%PYTHONPATH%"
)

if defined CLIP_SHAP_PYTHON (
    set "PYTHON_EXE=%CLIP_SHAP_PYTHON%"
) else if exist "%LEGACY_PYTHON%" (
    set "PYTHON_EXE=%LEGACY_PYTHON%"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" "%EXPERIMENT_DIR%src\run_clip_text_shap.py" %*
exit /b %ERRORLEVEL%
