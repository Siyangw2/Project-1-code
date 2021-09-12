########################################################
# Visualized analysis of the prediction results and drawn a line chart.
# Due to the large amount of forecast data, part of the forecast data is selected and displayed in the form of sub-pictures.
# Draw three subgraphs per line, and draw a total of two lines.
# Each subgraph shows 100 pieces of data.
########################################################
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('DT_model_3.csv')

# Draw a line chart of the prediction of the decision tree regression model
plt.figure(figsize=(18, 9))
for i in range(6):
    plt.subplot(321+i)
    plt.plot(data.loc[i*100:100+i*100, ['true']], label='true', c='orange')
    plt.plot(data.loc[i*100:100+i*100, ['pred']], label='pred', c='green')
    plt.xlabel('Time', size=16)
    plt.ylabel('total amount', size=16)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.savefig('DT_model_pred.png')
plt.show()

# Draw a line graph of the predictions of the two models
data_1 = pd.read_csv('LR_model_3.csv')
plt.figure(figsize=(18, 9))
for i in range(6):
    plt.subplot(321+i)
    plt.plot(data.loc[i*100:100+i*100, ['true']], label='true')
    plt.plot(data.loc[i*100:100+i*100, ['pred']], label='lr_pred', c='orange')
    plt.plot(data_1.loc[i*100:100+i*100, ['pred']], label='dt_pred', c='green')
    plt.xlabel('Time', size=16)
    plt.ylabel('total amount', size=16)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.legend()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.savefig('DT_LR_pred.png')
plt.show()