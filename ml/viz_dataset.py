import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')
print(df)

# df.drop(['platform_x'], axis=1, inplace=True)

sns.set()
sns.pairplot(df, hue='command', diag_kind='hist')
plt.show()
plt.savefig('pairplot.png')
#
# sns.set()
# sns.relplot(x='ball_x', y='ball_y', hue='command', data=df)
# plt.show()
