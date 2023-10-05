import yaml

data = { 'train' : '/Assault_Detection_Program/ADP_dataset2/train/images/',
         'val' : '/Assault_Detection_Program/ADP_dataset2/valid/images/',
         'test' : '/Assault_Detection_Program/ADP_dataset2/test/images/',
         'names' : ['gun', 'knife', 'person'],
         'nc' : 3 }

with open('/Assault_Detection_Program/ADP_dataset2/ADP_dataset2.yaml', 'w') as f:
    yaml.dump(data, f)

with open('/Assault_Detection_Program/ADP_dataset2/ADP_dataset2.yaml', 'r') as f:
    adp_yaml = yaml.safe_load(f)
    print(adp_yaml)