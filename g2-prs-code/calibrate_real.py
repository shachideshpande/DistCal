import pdb
#pdb.set_trace()
from scipy.interpolate import pchip_interpolate
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
#from matplotlib import pyplot as plt
from scipy.stats import linregress

import torch.nn as nn
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('fname', type=str, help='split')
parser.add_argument('save_file', type=str, help='save file')
parser.add_argument('norm', type=int, help='normalize') # 1 for True, 0 for False
args = parser.parse_args()
if(args.norm==1):
    args.norm=True
else:
    args.norm=False
#project_dir = "/Users/shachideshpande/Downloads/PRS_Data/"
project_dir = "/share/kuleshov/ssd86/UKBB_Downloads/calibration_scripts/prs-uncertainty"
NUM_BUCKETS=10

class Recalibrator(nn.Module):
  def __init__(self):
    super(Recalibrator, self).__init__()
    # self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.ModuleList([nn.Sequential(
                              nn.Linear(NUM_BUCKETS, 50),
                              nn.ReLU(),
                              nn.Linear(50, 2),
                              )]*26)


  def forward(self, inputs, targets):
    tau, quantiles = inputs
    mu, sigma = self.predict(inputs)
    return check_score(tau, mu, targets).mean()
    #loss = -1 * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
    #return loss.mean()
  def predict(self, inputs):
    # pdb.set_trace()
    tau, quantiles = inputs
    # x = self.flatten(x)
    outcome = self.linear_relu_stack[int(NUM_BUCKETS*tau)](quantiles)
    mu, logsigma = outcome[:, 0], outcome[:, 1]
    sigma = torch.exp(logsigma)
    return mu, sigma

class Recalibrator2(nn.Module):
  def __init__(self):
    super(Recalibrator2, self).__init__()
    # self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
                              nn.Linear(NUM_BUCKETS+1, 20),
                              nn.ReLU(),
                              nn.Linear(20, 2),
                              )


  def forward(self, inputs, targets):
    tau, quantiles = inputs
    mu, sigma = self.predict(inputs)
    return check_score(tau, mu, targets).mean()
    #loss = -1 * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
    #return loss.mean()
  def predict(self, inputs):
    # pdb.set_trace()
    tau, quantiles = inputs
    # x = self.flatten(x)
    inputs = torch.cat((torch.ones(quantiles.shape[0], 1)*tau, quantiles), 1)
    outcome = self.linear_relu_stack(inputs)
    mu, logsigma = outcome[:, 0], outcome[:, 1]
    sigma = torch.exp(logsigma)
    return mu, sigma

def check_score(tau, inverse_cdf, y):
  # check the check score computation f 
  # inverse_cdf = model(tau, quantiles)
  # pdb.set_trace()
  selector = (inverse_cdf>y.squeeze()).int()

  return (selector*(inverse_cdf - y.squeeze())*(1-tau) + (1-selector)*(y.squeeze() - inverse_cdf)*(tau))

class EarlyStopper:
  def __init__(self, patience=1, min_delta=0):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation_loss = float('inf')

  def early_stop(self, validation_loss):
    if validation_loss < self.min_validation_loss:
      self.min_validation_loss = validation_loss
      self.counter = 0
    elif validation_loss > (self.min_validation_loss + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False

def calibration_plot1(x_data, y_data, test=False, plot=False, print_log=False):
  # pdb.set_trace()
  expected_tau = np.linspace(0, 1, num=NUM_BUCKETS)
  empirical_tau = np.zeros(NUM_BUCKETS)
  empirical_tau_old = np.zeros(NUM_BUCKETS)
  for tau_i, tau in enumerate(expected_tau):
    q1 = model.predict(((1-tau)/2, x_data))[0]
    q2 = model.predict(((1+tau)/2, x_data))[0]

    if(test==1):
      q1_old = np.quantile(posterior_samples.astype(float), q=(1-tau)/2, axis=1).T[SUBSET:]
      q2_old = np.quantile(posterior_samples.astype(float), q=(1+tau)/2, axis=1).T[SUBSET:]
    elif(test==0):
      q1_old = np.quantile(posterior_samples.astype(float), q=(1-tau)/2, axis=1).T[:SPLIT]
      q2_old = np.quantile(posterior_samples.astype(float), q=(1+tau)/2, axis=1).T[:SPLIT]
    elif(test==2):
      q1_old = np.quantile(posterior_samples.astype(float), q=(1-tau)/2, axis=1).T[SPLIT:SUBSET]
      q2_old = np.quantile(posterior_samples.astype(float), q=(1+tau)/2, axis=1).T[SPLIT:SUBSET]
    elif(test==3):
      q1_old = np.quantile(posterior_samples_gwas.astype(float), q=(1-tau)/2, axis=1).T
      q2_old = np.quantile(posterior_samples_gwas.astype(float), q=(1+tau)/2, axis=1).T
      
    # pdb.set_trace()
    for outcome_i, outcome in enumerate(y_data):

      if(outcome>=q1[outcome_i] and outcome<=q2[outcome_i]):
        empirical_tau[tau_i]+=1
      if(outcome>=q1_old[outcome_i] and outcome<=q2_old[outcome_i]):
        empirical_tau_old[tau_i]+=1
    empirical_tau[tau_i]/=len(y_data)
    empirical_tau_old[tau_i]/=len(y_data)
    # if(tau_i>10):
    #   plt.plot(q1.detach().numpy(), label="q1")
    #   plt.plot(q2.detach().numpy(), label="q2")
    #   plt.plot(q1_old, label="q1_old")
    #   plt.plot(q2_old, label="q2_old")
    #   plt.plot(y_data.detach().numpy(), label="y_data")
    #   plt.legend()
    #   if(test):
    #     plt.savefig('/Users/shachideshpande/Downloads/PRS_Data/plot_results/chr22-new-test.png')
    #   else:

    #     plt.savefig('/Users/shachideshpande/Downloads/PRS_Data/plot_results/chr22-new-train.png')
    #   plt.clf()
  
  empirical_tau[len(empirical_tau)-1]=1
  empirical_tau_old[len(empirical_tau)-1]=1
  
  cal_score_before = ((empirical_tau_old-expected_tau)**2).sum()
  cal_score_after = ((empirical_tau-expected_tau)**2).sum()
  #return (cal_score_before, cal_score_after) 
  if(print_log):
    #  print((cal_score_before, cal_score_after))
    print(("Cal scores before and after Test=?"+str(test), cal_score_before, cal_score_after))
  if(plot):
  
    # print(expected_tau)

    # print(empirical_tau)

    # print(empirical_tau_old)
    # pdb.set_trace()
    
    #save_result = np.concatenate((expected_tau.reshape(-1, 1), empirical_tau.reshape(-1, 1), empirical_tau_old.reshape(-1, 1)), axis=1)
    #plt.plot(expected_tau, empirical_tau, label="calibrated")
    #plt.plot(expected_tau, empirical_tau_old, label="uncalibrated")
    #plt.axis('square')

    #plt.legend()
    if(test==1):
      #plt.savefig(project_dir+"plot_results/chr22-new-test."+fname+"_"+str(split)+".png")
      expected_tau = expected_tau.reshape((1,-1))
      empirical_tau = empirical_tau.reshape((1,-1))
      empirical_tau_old = empirical_tau_old.reshape((1,-1))
      save_taus = np.concatenate((expected_tau, empirical_tau, empirical_tau_old), axis=0)
      np.savetxt(project_dir+"/plot_results/adjusted.test."+fname+"_"+str(split)+".csv", save_taus, delimiter=",")
    elif(test==0):

      #plt.savefig(project_dir+"plot_results/chr22-new-train."+fname+"_"+str(split)+".png")
      expected_tau = expected_tau.reshape((1,-1))
      empirical_tau = empirical_tau.reshape((1,-1))
      empirical_tau_old = empirical_tau_old.reshape((1,-1))
      save_taus = np.concatenate((expected_tau, empirical_tau, empirical_tau_old), axis=0)
      np.savetxt(project_dir+"/plot_results/adjusted.train."+fname+"_"+str(split)+".csv", save_taus, delimiter=",")
    elif(test==2):

      #plt.savefig(project_dir+"plot_results/chr22-new-train."+fname+"_"+str(split)+".png")
      expected_tau = expected_tau.reshape((1,-1))
      empirical_tau = empirical_tau.reshape((1,-1))
      empirical_tau_old = empirical_tau_old.reshape((1,-1))
      save_taus = np.concatenate((expected_tau, empirical_tau, empirical_tau_old), axis=0)
      np.savetxt(project_dir+"/plot_results/adjusted.val."+fname+"_"+str(split)+".csv", save_taus, delimiter=",")

    elif(test==3):

      #plt.savefig(project_dir+"plot_results/chr22-new-train."+fname+"_"+str(split)+".png")
      expected_tau = expected_tau.reshape((1,-1))
      empirical_tau = empirical_tau.reshape((1,-1))
      empirical_tau_old = empirical_tau_old.reshape((1,-1))
      save_taus = np.concatenate((expected_tau, empirical_tau, empirical_tau_old), axis=0)
      np.savetxt(project_dir+"/plot_results/adjusted.gwas."+fname+"_"+str(split)+".csv", save_taus, delimiter=",")

    #plt.clf()
  return (cal_score_before, cal_score_after)

def final_check_score(x_data, y_data, test=0, print_log=False):
  check_score_before, check_score_after = 0, 0
  # evaluate_taus=np.linspace
  for i, tau in enumerate(taus):
    temp_before = check_score(tau, x_data[:, i], y_data).mean()
    temp_after = check_score(tau, model.predict((tau, x_data))[0], y_data).mean()
    check_score_before += temp_before
    check_score_after += temp_after
    # print((temp_before, temp_after))
  if(test==1):
    prepend = "[TEST]"
  elif(test==0):
    prepend = "[TRAIN]"
  elif(test==2):
    prepend = "[VAL]"
  elif(test==3):
    prepend = "[GWAS]"
  
  # Old r^2 computation, not stochastic
  #rvalue_before = (stats.linregress(x_data[:, 12], y_data.squeeze()).rvalue)
  #rvalue_after = stats.linregress(model.predict((0.5, x_data))[0].detach().numpy(), y_data.squeeze()).rvalue
  x_before = np.array([np.linspace((x_data[x_i]).min(), (x_data[x_i]).max(), num=100) for x_i in range(len(x_data))])
  y_d_before = np.array([pchip_interpolate(x_data[x_i], np.linspace(0, 1, num=NUM_BUCKETS), x_before[x_i], der=1) for x_i in range(len(x_data)) ])
  y_d_before = np.array([y_d_before[x_i]/y_d_before[x_i].sum() for x_i in range(len(x_data)) ])
  y_d_before[y_d_before<0] = 0 # replacing occasional very small negative values with 0
  calib_means_before = np.array([np.random.choice(x_before[x_i], size=500, p=y_d_before[x_i]).mean() for x_i in range(len(x_data))])

  # rvalue_before = (stats.linregress(x_data[:, 12], y_data.squeeze()).rvalue)
  rvalue_before = (stats.linregress(calib_means_before, y_data.squeeze()).rvalue)


  calib_x_data =  np.array([model.predict((tau, x_data))[0].detach().numpy() for tau in np.linspace(0, 1, num=NUM_BUCKETS)]).T

  
  

  # calib_x_data = np.sort(calib_x_data, axis=1)
  if((np.diff(calib_x_data)<=0).sum()>0):
    rvalue_after = -1
  else:
    x_after = np.array([np.linspace((calib_x_data[x_i]).min(), (calib_x_data[x_i]).max(), num=100) for x_i in range(len(calib_x_data))])
    try:
        y_d_after = np.array([pchip_interpolate(calib_x_data[x_i], np.linspace(0, 1, num=NUM_BUCKETS), x_after[x_i], der=1) for x_i in range(len(calib_x_data)) ])
    except:
        pdb.set_trace()
    y_d_after = np.array([y_d_after[x_i]/y_d_after[x_i].sum() for x_i in range(len(calib_x_data)) ])
    y_d_after[y_d_after<0] = 0 # replacing occasional very small negative values with 0
    calib_means_after = np.array([np.random.choice(x_after[x_i], size=500, p=y_d_after[x_i]).mean() for x_i in range(len(calib_x_data))])
    # rvalue_after = stats.linregress(model.predict((0.5, x_data))[0].detach().numpy(), y_data.squeeze()).rvalue
    rvalue_after = stats.linregress(calib_means_after, y_data.squeeze()).rvalue
  if(print_log):
    print((prepend+" Check scores (before, after)=",check_score_before, check_score_after))
    print(prepend+" R^2 values (before, after)", rvalue_before, rvalue_after)
  return {"rvalue": (rvalue_before, rvalue_after), "check_score": (check_score_before, check_score_after.detach())}



final_results = {"rvalue_before":{"train":[], "test":[], "val":[], "gwas":[]},
"rvalue_after": {"train":[], "test":[], "val":[], "gwas":[]},
"check_score_before": {"train":[], "test":[], "val":[], "gwas":[]}, 
"check_score_after": {"train":[], "test":[], "val":[], "gwas":[]}, 
"cal_score_before": {"train":[], "test":[], "val":[], "gwas":[]},
"cal_score_after": {"train":[], "test":[], "val":[], "gwas":[]}
}
save_file = project_dir+"/cal_run_logs/"+args.save_file+args.fname+"_adjusted.pkl"
log_file = project_dir+"/cal_run_logs/run_logs."+args.save_file+args.fname+"_adjusted.pkl"
normalize=args.norm
#normalize=False
print("normalize", normalize)
fname = args.fname
log_dict=dict()
reps = 5
log_dict["rvalue_before"]={"train":[[]  for i in range(reps)], "val":[[] for i in range(reps)], "test":[[] for i in range(reps)]}
log_dict["rvalue_after"]={"train":[[]  for i in range(reps)], "val":[[] for i in range(reps)], "test":[[] for i in range(reps)]}
log_dict["check_score_before"]={"train":[[]  for i in range(reps)], "val":[[] for i in range(reps)], "test":[[] for i in range(reps)]}
log_dict["check_score_after"]={"train":[[]  for i in range(reps)], "val":[[] for i in range(reps)], "test":[[] for i in range(reps)]}
log_dict["cal_score_before"]={"train":[[]  for i in range(reps)], "val":[[] for i in range(reps)], "test":[[] for i in range(reps)]}
log_dict["cal_score_after"]={"train":[[]  for i in range(reps)], "val":[[] for i in range(reps)], "test":[[] for i in range(reps)]}
for iteration in range(5):
    split = iteration+1

    SUBSET = 1000
    #pdb.set_trace()

    #posterior_samples = pd.read_csv('/Users/shachideshpande/Downloads/PRS_Data/plain_posterior_gv_samples_chr22.785.hsq.'+HERIT+'.'+LDSET+'.csv').to_numpy() #[:, :-15]


    posterior_samples = pd.read_csv(project_dir+"/real-data/test_posterior_samples/"+fname+"_"+str(split)+"_1.csv").to_numpy() #[:, :-15]
    posterior_samples_gwas = pd.read_csv(project_dir+"/real-data/train_posterior_samples/"+fname+"_"+str(split)+"_1.csv").to_numpy()
    # posterior_samples.mean(axis=1) for mean outcome - but calibrated quantile needs to be inverted for estimating means
    #
    if(normalize):
        posterior_samples = posterior_samples - posterior_samples.mean(axis=0)
        posterior_samples_gwas = posterior_samples_gwas - posterior_samples_gwas.mean(axis=0)

    X = (np.quantile(posterior_samples.astype(float), q=np.linspace(0, 1, num=NUM_BUCKETS), axis=1).T)
    X[:, 0] = posterior_samples.mean(axis=1)-6*posterior_samples.std(axis=1)
    X[:, -1] = posterior_samples.mean(axis=1)+6*posterior_samples.std(axis=1)

    X_gwas = (np.quantile(posterior_samples_gwas.astype(float), q=np.linspace(0, 1, num=NUM_BUCKETS), axis=1).T)
    X_gwas[:, 0] = posterior_samples_gwas.mean(axis=1)-6*posterior_samples_gwas.std(axis=1)
    X_gwas[:, -1] = posterior_samples_gwas.mean(axis=1)+6*posterior_samples_gwas.std(axis=1)
    
    #pdb.set_trace()
    X_orig = posterior_samples.astype(float)
# sprintf("/share/kuleshov/ssd86/UKBB_Downloads/calibration_scripts/prs-uncertainty/real_pheno_splits/train_%s_adjusted%d.csv", pheno_name, SEED)    
    #y = pd.read_csv(project_dir+"/real_pheno_splits/"+fname+"_split"+str(split)+"_test.csv", header=None, sep='\t').to_numpy()[:2000, 2]
    #y_gwas = pd.read_csv(project_dir+"/real_pheno_splits/"+fname+"_split"+str(split)+"_.csv", header=None, sep='\t').to_numpy()[:1000, 2]


    y = pd.read_csv(project_dir+"/real_pheno_splits/test_"+fname+"_adjusted"+str(split)+".csv", sep='\t').to_numpy().squeeze()
    y_gwas = pd.read_csv(project_dir+"/real_pheno_splits/train_"+fname+"_adjusted"+str(split)+".csv", sep='\t').to_numpy().squeeze()
    #pdb.set_trace()
    X = X[np.logical_not(np.isnan(y))]
    X_gwas = X_gwas[np.logical_not(np.isnan(y_gwas))]
    X_orig = X_orig[np.logical_not(np.isnan(y))]

    y = y[np.logical_not(np.isnan(y))]
    y_gwas = y_gwas[np.logical_not(np.isnan(y_gwas))]

    #y = posterior_samples.mean(axis=1)
    #y_gwas = posterior_samples_gwas.mean(axis=1)
    #pdb.set_trace()
    slope, intercept, r, p, se = linregress(X_orig.mean(axis=1), y.squeeze())
    #y = y

    y_mean = np.mean(y)
    y_std = 1*np.std(y)
    #y = (y-y_mean)/(y_std)
    print((y_mean, y_std))

# pdb.set_trace()

#y= pd.read_csv('/Users/shachideshpande/Downloads/PRS_Data/posterior_gvs_chr22.csv').to_numpy()


    validity = np.array([int(X[i][0]<=y_i and X[i][-1]>=y_i) for i, y_i in enumerate(y)])
    print(validity.mean())

# pdb.set_trace()
    SPLIT = int(0.8*SUBSET)
    X_train = X[:SPLIT]
    y_train = y[:SPLIT]
    X_val = X[SPLIT:SUBSET]
    y_val = y[SPLIT:SUBSET]
    X_test = X[SUBSET:]
    y_test = y[SUBSET:]
    X_train = torch.from_numpy(X_train).to(torch.float32)
    y_train = torch.from_numpy(y_train).to(torch.float32)
    X_val = torch.from_numpy(X_val).to(torch.float32)
    y_val = torch.from_numpy(y_val).to(torch.float32)
    X_test = torch.from_numpy(X_test).to(torch.float32)
    y_test = torch.from_numpy(y_test).to(torch.float32)
    X_gwas = torch.from_numpy(X_gwas).to(torch.float32)
    y_gwas = torch.from_numpy(y_gwas).to(torch.float32)
    taus = np.linspace(0, 1, num=NUM_BUCKETS)
    model = Recalibrator2()

    final_check_score(X_train, y_train)
    final_check_score(X_test, y_test)

    # exit(1)
    if(normalize):
        y_test = y_test - y_test.mean()
        y_val = y_val - y_val.mean()
        y_train = y_train - y_train.mean()
        y_gwas = y_gwas - y_gwas.mean()


    # regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)




    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # loss_fn = CustomLoss()
    num_epochs=1500
    early_stopper = EarlyStopper(patience=50, min_delta=0.0005)
    for epoch in range(num_epochs):
        # 
        # Forward pass
        # pdb.set_trace()
        for tau in taus:

          loss = model((tau, X_train), y_train)
        # loss = loss_fn(outputs, targets)

        # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        if(epoch%1==0): # 1 when collecting logs
          print("Epoch "+str(epoch))

          cal_before_train, cal_after_train  = calibration_plot1(X_train, y_train, test=0,print_log=True)
          cal_before_test, cal_after_test = calibration_plot1(X_test, y_test, test=1, print_log=True)
          cal_before_val, cal_after_val = calibration_plot1(X_val, y_val, test=2, print_log=True)
          res_train = final_check_score(X_train, y_train, test=0, print_log=True)
          res_test  = final_check_score(X_test, y_test, test=1, print_log=True)
          res_val = final_check_score(X_val, y_val, test=2, print_log=True)
          rvalue_train, check_score_train = res_train["rvalue"], res_train["check_score"]
          rvalue_test, check_score_test = res_test["rvalue"], res_test["check_score"]
          rvalue_val, check_score_val = res_val["rvalue"], res_val["check_score"]
          log_dict["rvalue_before"]["train"][iteration].append(rvalue_train[0])
          log_dict["rvalue_before"]["val"][iteration].append(rvalue_val[0])
          log_dict["rvalue_before"]["test"][iteration].append(rvalue_test[0])
          log_dict["rvalue_after"]["train"][iteration].append(rvalue_train[1])
          log_dict["rvalue_after"]["val"][iteration].append(rvalue_val[1])
          log_dict["rvalue_after"]["test"][iteration].append(rvalue_test[1])
          log_dict["check_score_before"]["train"][iteration].append(check_score_train[0])
          log_dict["check_score_before"]["val"][iteration].append(check_score_val[0])
          log_dict["check_score_before"]["test"][iteration].append(check_score_test[0])
          log_dict["check_score_after"]["train"][iteration].append(check_score_train[1])
          log_dict["check_score_after"]["val"][iteration].append(check_score_val[1])
          log_dict["check_score_after"]["test"][iteration].append(check_score_test[1])
          log_dict["cal_score_before"]["train"][iteration].append(cal_before_train)
          log_dict["cal_score_before"]["val"][iteration].append(cal_before_val)
          log_dict["cal_score_before"]["test"][iteration].append(cal_before_test)
          log_dict["cal_score_after"]["train"][iteration].append(cal_after_train)
          log_dict["cal_score_after"]["val"][iteration].append(cal_after_val)
          log_dict["cal_score_after"]["test"][iteration].append(cal_after_test)
        cal_before_val, cal_after_val = calibration_plot1(X_val, y_val, test=2)

        #validation_loss = final_check_score(X_val, y_val, test=2)["check_score"][1]
         
        if early_stopper.early_stop(cal_after_val):
          break
    #pdb.set_trace()
    cal_before_train, cal_after_train = calibration_plot1(X_train, y_train, test=0, plot=True,print_log=True)
    cal_before_test, cal_after_test = calibration_plot1(X_test, y_test, test=1, plot=True, print_log=True)
    cal_before_val, cal_after_val = calibration_plot1(X_val, y_val, test=2, plot=True, print_log=True)
    cal_before_gwas, cal_after_gwas = calibration_plot1(X_gwas, y_gwas, test=3, plot=True, print_log=True)
    
    res_train = final_check_score(X_train, y_train, print_log=True)
    res_test  = final_check_score(X_test, y_test, test=1,  print_log=True)
    res_val = final_check_score(X_val, y_val, test=2,  print_log=True)
    res_gwas = final_check_score(X_gwas, y_gwas, test=3,  print_log=True)

    rvalue_train, check_score_train = res_train["rvalue"], res_train["check_score"]
    rvalue_test, check_score_test = res_test["rvalue"], res_test["check_score"]
    rvalue_val, check_score_val = res_val["rvalue"], res_val["check_score"]
    rvalue_gwas, check_score_gwas = res_gwas["rvalue"], res_gwas["check_score"]

    #final_results = {"rvalue_before":{"train":[], "test":[], "val":[], "gwas":[]}},
    #    "rvalue_after": {"train":[], "test":[], "val":[], "gwas":[]},
    #    "check_score_before": {"train":[], "test":[], "val":[], "gwas":[]},
    #    "check_score_after": {"train":[], "test":[], "val":[], "gwas":[]},
    #    "cal_score_before": {"train":[], "test":[], "val":[], "gwas":[]},
    #    "cal_score_after": {"train":[], "test":[], "val":[], "gwas":[]}
    #    }
    final_results["rvalue_before"]["train"].append(rvalue_train[0])
    final_results["rvalue_before"]["val"].append(rvalue_val[0])
    final_results["rvalue_before"]["test"].append(rvalue_test[0])
    final_results["rvalue_before"]["gwas"].append(rvalue_gwas[0])

    final_results["rvalue_after"]["train"].append(rvalue_train[1])
    final_results["rvalue_after"]["val"].append(rvalue_val[1])
    final_results["rvalue_after"]["test"].append(rvalue_test[1])
    final_results["rvalue_after"]["gwas"].append(rvalue_gwas[1])

    final_results["check_score_before"]["train"].append(check_score_train[0])
    final_results["check_score_before"]["val"].append(check_score_val[0])
    final_results["check_score_before"]["test"].append(check_score_test[0])
    final_results["check_score_before"]["gwas"].append(check_score_gwas[0])

    final_results["check_score_after"]["train"].append(check_score_train[1])
    final_results["check_score_after"]["val"].append(check_score_val[1])
    final_results["check_score_after"]["test"].append(check_score_test[1])
    final_results["check_score_after"]["gwas"].append(check_score_gwas[1])

    final_results["cal_score_before"]["train"].append(cal_before_train)
    final_results["cal_score_before"]["val"].append(cal_before_val)
    final_results["cal_score_before"]["test"].append(cal_before_test)
    final_results["cal_score_before"]["gwas"].append(cal_before_gwas)

    final_results["cal_score_after"]["train"].append(cal_after_train)
    final_results["cal_score_after"]["val"].append(cal_after_val)
    final_results["cal_score_after"]["test"].append(cal_after_test)
    final_results["cal_score_after"]["gwas"].append(cal_after_gwas)

import pickle 

#with open(save_file, 'wb') as f:
#    pickle.dump(final_results, f)
#pdb.set_trace()
pickle.dump(final_results, open(save_file, "wb"))
pickle.dump(log_dict, open(log_file, "wb"))
#with open(save_file, 'rb') as f:
#    final_results = pickle.load(f)
#pdb.set_trace()
print("==================================")
print("Final metrics")
print("==================================")
print("Dataset", "Train", "Test", "Val", "GWAS", sep="\t")
print("Rvalue before [Mean]", np.array(final_results["rvalue_before"]["train"]).mean(), np.array(final_results["rvalue_before"]["test"]).mean(), np.array(final_results["rvalue_before"]["val"]).mean(), np.array(final_results["rvalue_before"]["gwas"]).mean(), sep="\t")
print("Rvalue before [Std error]", np.array(final_results["rvalue_before"]["train"]).std()/np.sqrt(5), np.array(final_results["rvalue_before"]["test"]).std()/np.sqrt(5), np.array(final_results["rvalue_before"]["val"]).std()/np.sqrt(5), np.array(final_results["rvalue_before"]["gwas"]).std()/np.sqrt(5), sep="\t")


print("Rvalue after [Mean]", np.array(final_results["rvalue_after"]["train"]).mean(), np.array(final_results["rvalue_after"]["test"]).mean(), np.array(final_results["rvalue_after"]["val"]).mean(), np.array(final_results["rvalue_after"]["gwas"]).mean(), sep="\t")
print("Rvalue after [Std error]", np.array(final_results["rvalue_after"]["train"]).std()/np.sqrt(5), np.array(final_results["rvalue_after"]["test"]).std()/np.sqrt(5), np.array(final_results["rvalue_after"]["val"]).std()/np.sqrt(5), np.array(final_results["rvalue_after"]["gwas"]).std()/np.sqrt(5), sep="\t")

print("Check score before [Mean] ", np.array(final_results["check_score_before"]["train"]).mean(), np.array(final_results["check_score_before"]["test"]).mean(), np.array(final_results["check_score_before"]["val"]).mean(), np.array(final_results["check_score_before"]["gwas"]).mean(), sep='\t')
print("Check score before [Std error]", np.array(final_results["check_score_before"]["train"]).std()/np.sqrt(5), np.array(final_results["check_score_before"]["test"]).std()/np.sqrt(5), np.array(final_results["check_score_before"]["val"]).std()/np.sqrt(5), np.array(final_results["check_score_before"]["gwas"]).std()/np.sqrt(5), sep='\t')

print("Check score after [Mean]", np.array(final_results["check_score_after"]["train"]).mean(), np.array(final_results["check_score_after"]["test"]).mean(), np.array(final_results["check_score_after"]["val"]).mean(), np.array(final_results["check_score_after"]["gwas"]).mean(), sep='\t')
print("Check score after [Std error]", np.array(final_results["check_score_after"]["train"]).std()/np.sqrt(5), np.array(final_results["check_score_after"]["test"]).std()/np.sqrt(5), np.array(final_results["check_score_after"]["val"]).std()/np.sqrt(5), np.array(final_results["check_score_after"]["gwas"]).std()/np.sqrt(5), sep='\t')

print("Cal score before [Mean]",np.array(final_results["cal_score_before"]["train"]).mean(), np.array(final_results["cal_score_before"]["test"]).mean(), np.array(final_results["cal_score_before"]["val"]).mean(), np.array(final_results["cal_score_before"]["gwas"]).mean(),  sep='\t')
print("Cal score before [Std error]",np.array(final_results["cal_score_before"]["train"]).std()/np.sqrt(5), np.array(final_results["cal_score_before"]["test"]).std()/np.sqrt(5), np.array(final_results["cal_score_before"]["val"]).std()/np.sqrt(5), np.array(final_results["cal_score_before"]["gwas"]).std()/np.sqrt(5),  sep='\t')

print("Cal score after [Mean]", np.array(final_results["cal_score_after"]["train"]).mean(), np.array(final_results["cal_score_after"]["test"]).mean(), np.array(final_results["cal_score_after"]["val"]).mean(), np.array(final_results["cal_score_after"]["gwas"]).mean(), sep='\t')
print("Cal score after [Std error]", np.array(final_results["cal_score_after"]["train"]).std()/np.sqrt(5), np.array(final_results["cal_score_after"]["test"]).std()/np.sqrt(5), np.array(final_results["cal_score_after"]["val"]).std()/np.sqrt(5), np.array(final_results["cal_score_after"]["gwas"]).std()/np.sqrt(5), sep='\t')
