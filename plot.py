import numpy as np
import matplotlib.pyplot as plt

# plot the execution time of resnet56

def plot_resnet56():
    time_cpu_rec_base = np.loadtxt('time_record/cpu_time_record_res56_base')
    time_real_rec_base = np.loadtxt('time_record/real_time_record_res56_base')
    time_cpu_rec_1 = np.loadtxt('time_record/cpu_time_record_res56_1')
    time_real_rec_1 = np.loadtxt('time_record/real_time_record_res56_1')
    time_cpu_rec = np.loadtxt('time_record/cpu_time_record_res56')
    time_real_rec = np.loadtxt('time_record/real_time_record_res56')
    
    x = np.zeros(7)
    for i in range(10,150,20):
        x[int(i/20)] = i
    plt.plot(time_cpu_rec,'b')
    plt.plot(time_real_rec,'r')
    plt.plot(time_cpu_rec_1,'g')
    plt.plot(time_real_rec_1,'m')
    plt.plot(time_cpu_rec_base,'c')
    plt.plot(time_real_rec_base,'y')
    plt.xlabel("batch size")
    plt.ylabel("time/sec")
    plt.title("ResNet56 compare/batch")
    plt.ylim([-3,64])
    plt.xticks(x)
    plt.legend(["cpu time res56", "real time res56", "cpu time res56_1", "real time res56_1", "cpu time res56_base", "real time res56_base"], loc ="upper right") 
    plt.savefig("time_record/time_record_res56_compare.jpg")
    plt.show()

# plot the execution time of resnet110

def plot_resnet110():
    time_cpu_rec_base = np.loadtxt('time_record/cpu_time_record_res110_base')
    time_real_rec_base = np.loadtxt('time_record/real_time_record_res110_base')
    time_cpu_rec_1 = np.loadtxt('time_record/cpu_time_record_res110_1')
    time_real_rec_1 = np.loadtxt('time_record/real_time_record_res110_1')
    time_cpu_rec = np.loadtxt('time_record/cpu_time_record_res110')
    time_real_rec = np.loadtxt('time_record/real_time_record_res110')
    x = np.zeros(7)
    for i in range(10,150,20):
        x[int(i/20)] = i
    plt.plot(time_cpu_rec,'b')
    plt.plot(time_real_rec,'r')
    plt.plot(time_cpu_rec_1,'g')
    plt.plot(time_real_rec_1,'m')
    plt.plot(time_cpu_rec_base,'c')
    plt.plot(time_real_rec_base,'y')
    plt.xlabel("batch size")
    plt.ylabel("time/sec")
    plt.title("ResNet110 real time/batch")
    plt.ylim([-3,100])
    plt.xticks(x)
    plt.legend(["cpu time res110", "real time res110", "cpu time res110_1", "real time res110_1", "cpu time res110_base", "real time res110_base"], loc ="upper right")
    plt.savefig("time_record/time_record_res110_real_time.jpg")
    plt.show()

plot_resnet56()
plot_resnet110()