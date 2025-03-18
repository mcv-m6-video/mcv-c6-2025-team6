import os

def get_next_experiment_folder(base_output_dir, name="predict"):
    os.makedirs(base_output_dir, exist_ok=True)
    
    existing_exps = [d for d in os.listdir(base_output_dir) if d.startswith(f"_{name}")]
    
    exp_numbers = [int(d.replace(f"_{name}", "")) for d in existing_exps if d.replace(f"_{name}", "").isdigit()]

    next_exp = max(exp_numbers, default=0) + 1
    
    return os.path.join(base_output_dir, f"_{name}{next_exp}")