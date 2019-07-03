from torchvision import datasets, models, transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train_v1': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    'train_v2': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    'train_v3': transforms.Compose([
            #transforms.Resize(80),
            #transforms.RandomAffine(degrees=[-8,8]), #, shear=30),
            #transforms.RandomAffine(degrees=0,scale=[0.1,0.3]), #, shear=30),
            #transforms.RandomAffine(degrees=0, shear=30),
            #transforms.CenterCrop(250),
            transforms.RandomResizedCrop(224, scale=(0.5, 1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    'train_v4': transforms.Compose([
            # transforms.Resize(290),
            # transforms.RandomAffine(degrees=0, shear=10),
            # transforms.CenterCrop(250),
            transforms.RandomResizedCrop(224, scale=(0.4, 1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    'val': transforms.Compose([
            transforms.Resize(180),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            normalize,
        ]),
    'val_v4': transforms.Compose([
            # transforms.Resize(290),
            # transforms.RandomAffine(degrees=0, shear=10),
            # transforms.CenterCrop(250),
            transforms.RandomResizedCrop(224, scale=(0.55, 1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    'val_v3': transforms.Compose([
            # transforms.Resize(290),
            # transforms.RandomAffine(degrees=0, shear=10),
            # transforms.CenterCrop(250),
            transforms.RandomResizedCrop(224, scale=(0.55, 1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    'val_em': transforms.Compose([
            # transforms.Resize(290),
            # transforms.RandomAffine(degrees=0, shear=10),
            # transforms.CenterCrop(250),
            transforms.RandomResizedCrop(224, scale=(0.7, 1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    'test_v1': transforms.Compose([
            transforms.Resize(180),
            transforms.TenCrop(160),
            transforms.ToTensor(),
            normalize,
        ]),
    'test_v2': transforms.Compose([
            transforms.Resize(250),
            transforms.TenCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
    'rtest_v1': transforms.Compose([
            transforms.Resize(180),
            transforms.TenCrop(160),
            transforms.ToTensor(),
            normalize,
        ]),
    'rtest_v2': transforms.Compose([
            transforms.Resize(250),
            transforms.TenCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
}
data_transforms_old = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomAffine(degrees=360, scale=(0.7, 1.3), shear=30), #fillcolor=128
        transforms.RandomCrop(196),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_v2': transforms.Compose([
            transforms.Resize(400),
            transforms.RandomAffine(degrees=360, shear=30),
            transforms.CenterCrop(280),
            transforms.RandomResizedCrop(196,scale=(0.5,1)),
            transforms.ColorJitter(brightness=0.3,contrast=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(196),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_v2': transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(196),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_mask': transforms.Compose([
            transforms.RandomAffine(degrees=360), #, shear=30),
            transforms.RandomResizedCrop(196,scale=(0.5,1)),
            transforms.ColorJitter(brightness=0.3,contrast=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

    'val_mask': transforms.Compose([
        transforms.CenterCrop(196),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': [
                transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(196),  #change FiveCrop
                 ]),
                transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(196),
                 ]),
            ],
    'test_v2': [
            transforms.Compose([
            transforms.Resize(400),
            transforms.CenterCrop(196),  #change FiveCrop
             ]),
            transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
            transforms.Resize(400),
            transforms.RandomAffine(degrees=360),
            transforms.CenterCrop(280),
            transforms.RandomResizedCrop(196,scale = (0.7,1)),
             ]),
            ],
    'test_mask': [
            transforms.Compose([
            transforms.CenterCrop(196),  #change FiveCrop
             ]),
            transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
            transforms.RandomAffine(degrees=360),
            transforms.RandomResizedCrop(196,scale = (0.7,1)),
             ]),
            ]

}
