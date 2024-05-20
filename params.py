# 训练
epochs = 20
batch_size = 2
num_workers = 4
train_data = "D:/算法汇总/遥感/训练数据/地物分类/建筑/西安/切片1/train"
val_data = "D:/算法汇总/遥感/训练数据/地物分类/建筑/西安/切片1/val"
save_data_path = "D:/算法汇总/遥感/地物分类/Land Cover Classification - bak/models"

# 预测

# 星图地球
# url = 'https://tiles1.geovisearth.com/base/v1/img/{z}/{x}/{y}?format=webp&tmsIds=w&token=0aeb02f29320b060c2e2d0c04eb4887c6b8d5a8ed479b3aacff2b6a273b0d38d'

# esri影像
url = 'https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'

# 天地图影像
# url = 'https://t3.tianditu.gov.cn/img_w/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=img&STYLE=default&TILEMATRIXSET=w&FORMAT=tiles&TileMatrix={z}&TileRow={y}&TileCol={x}&tk=4267820f43926eaf808d61dc07269beb'

# 高德影像
# url = 'http://wprd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=6&x={x}&y={y}&z={z}'
min_lon,max_lat = 108.988348  ,34.223430
max_lon,min_lat = 109.000750  ,34.208297
zoom_level = 18
result = r'D:\image\ceshi'
image = r"D:\image\tif4.tif"
model_path = "models/epoch_59_acc_0.87429_kappa_0.70562.pth"