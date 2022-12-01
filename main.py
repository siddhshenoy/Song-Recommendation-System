import FeaturePlots

# print(f"Data Pre-Processing's path: {DataPreProcessing.csv_path}")
# print(f"Data Pre-Processing's files: {DataPreProcessing.csv_files}")

# df = DataPreProcessing.DataPreProcessing().return_data_frame()

# print(df['tempo'].min(), df['tempo'].max())

path = FeaturePlots.csv_path
files = FeaturePlots.csv_files


FeaturePlots.FeaturePlots(path=path, files=files, y=0, plot3D=True)
