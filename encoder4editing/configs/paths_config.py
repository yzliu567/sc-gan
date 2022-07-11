dataset_paths = {
	#  Face Datasets (In the paper: FFHQ - train, CelebAHQ - test)
	'ffhq': '', # ffhq dataset path
	'celeba_test': '', #celeba dataset path

	#  Cars Dataset (In the paper: Stanford cars)
	'cars_train': '',
	'cars_test': '',

	#  Horse Dataset (In the paper: LSUN Horse)
	'horse_train': '',
	'horse_test': '',

	#  Church Dataset (In the paper: LSUN Church)
	'church_train': '',
	'church_test': '',

	#  Cats Dataset (In the paper: LSUN Cat)
	'cats_train': '', #afhq cats dataset path
	'cats_test': '' #afhq cats dataset path
}

model_paths = {
	'stylegan_ffhq': '', #pretrained stylegan checkpoints path
	'ir_se50': './pretrained_models/model_ir_se50.pth',
	'shape_predictor': './pretrained_models/shape_predictor_68_face_landmarks.dat',
	'moco': './pretrained_models/moco_v2_800ep_pretrain.pt'
}
