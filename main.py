#main.py
'''
3/25/2026
Matthew Miller

ECG_project0/

'''

from ecg_lib.load_data0 import load_data0

class Main:         #I really don't need to make this a class, but it's practice
    def run(self):
        
        #import settings
        from pathlib import Path
        import tomllib
        
        #get settings.toml information
        base_dir = Path(__file__).resolve().parent
        settings_path = base_dir / "settings.toml"
        with open(settings_path, "rb") as f:
            config = tomllib.load(f)
    
        #load data
        self.data, self.truth = load_data0(config)

        #split data
        
        #call the training function in train.py



if __name__ == "__main__":
    Main().run()