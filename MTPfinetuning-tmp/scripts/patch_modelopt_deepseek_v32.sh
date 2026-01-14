#!/bin/bash
# =============================================================================
# 为 DeepSeek-V3.2 添加 ModelOpt 支持的补丁脚本
# 需要在所有节点上运行此脚本
# =============================================================================

set -e

echo "==================================================================="
echo "为 DeepSeek-V3.2 添加 ModelOpt 支持"
echo "==================================================================="

# 1. 添加 DeepseekV32ForCausalLM 到 mcore_common.py
python3 << 'PYEOF'
file_path = '/usr/local/lib/python3.12/dist-packages/modelopt/torch/export/plugins/mcore_common.py'

with open(file_path, 'r') as f:
    content = f.read()

if 'DeepseekV32ForCausalLM' in content:
    print("[mcore_common.py] 已包含 DeepseekV32ForCausalLM")
else:
    content = content.replace(
        '"DeepseekV3ForCausalLM": deepseek_causal_lm_export,',
        '"DeepseekV3ForCausalLM": deepseek_causal_lm_export,\n    "DeepseekV32ForCausalLM": deepseek_causal_lm_export,'
    )
    content = content.replace(
        '"DeepseekV3ForCausalLM": deepseek_causal_lm_import,',
        '"DeepseekV3ForCausalLM": deepseek_causal_lm_import,\n    "DeepseekV32ForCausalLM": deepseek_causal_lm_import,'
    )
    with open(file_path, 'w') as f:
        f.write(content)
    print("[mcore_common.py] 已添加 DeepseekV32ForCausalLM")
PYEOF

# 2. 修改 megatron_importer.py 处理 deepseek_v32 model_type
python3 << 'PYEOF'
file_path = '/usr/local/lib/python3.12/dist-packages/modelopt/torch/export/plugins/megatron_importer.py'

with open(file_path, 'r') as f:
    content = f.read()

if 'deepseek_v32' in content:
    print("[megatron_importer.py] 已包含 deepseek_v32 处理逻辑")
else:
    old_code = '''        """Create a GPTModel importer instance."""
        self._hf_config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True
        )'''
    
    new_code = '''        """Create a GPTModel importer instance."""
        # Handle DeepSeek-V3.2 which uses deepseek_v32 model_type not recognized by transformers
        config_class = None
        try:
            import json
            config_path = Path(pretrained_model_name_or_path) / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                if config_dict.get("model_type") == "deepseek_v32":
                    # DeepSeek-V3.2 uses same config structure as DeepSeek-V3
                    from transformers import DeepseekV3Config
                    config_class = DeepseekV3Config
        except Exception:
            pass
        
        if config_class is not None:
            self._hf_config = config_class.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )
        else:
            self._hf_config = transformers.AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(file_path, 'w') as f:
            f.write(content)
        print("[megatron_importer.py] 已添加 deepseek_v32 处理逻辑")
    else:
        print("[megatron_importer.py] 警告: 未找到要替换的代码块，可能格式不同")
PYEOF

# 3. 验证
python3 << 'PYEOF'
print("\n=== 验证补丁 ===")

# 检查 mcore_common.py
with open('/usr/local/lib/python3.12/dist-packages/modelopt/torch/export/plugins/mcore_common.py', 'r') as f:
    if 'DeepseekV32ForCausalLM' in f.read():
        print("✓ mcore_common.py: DeepseekV32ForCausalLM 已添加")
    else:
        print("✗ mcore_common.py: DeepseekV32ForCausalLM 未找到")

# 检查 megatron_importer.py
with open('/usr/local/lib/python3.12/dist-packages/modelopt/torch/export/plugins/megatron_importer.py', 'r') as f:
    if 'deepseek_v32' in f.read():
        print("✓ megatron_importer.py: deepseek_v32 处理逻辑已添加")
    else:
        print("✗ megatron_importer.py: deepseek_v32 处理逻辑未找到")
PYEOF

echo "==================================================================="
echo "补丁完成！"
echo "==================================================================="
