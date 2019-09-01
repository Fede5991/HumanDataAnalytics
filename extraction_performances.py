import numpy as np

def extraction_performances(validation_accuracy,training_accuracy,variables,iterations,attempts):
    performances=np.zeros((variables,iterations))
    performances2=np.zeros((variables,iterations))
    a = attempts
    for j in range(variables):
        k=0
        for l in range(a):
            for i in range(iterations):
                if (i%(a**(j+1)))<((l+1)*a**(j+1)/a) and (i%(a**(j+1)))>=(l*a**(j+1)/a):
                    performances[j,k]=validation_accuracy[i]
                    performances2[j,k]=training_accuracy[i]
                    k=k+1

    general_perf = np.zeros((variables,attempts))
    general_perf2 = np.zeros((variables,attempts))

    for i in range(variables):
        for j in range(attempts):
            general_perf[i,j]=np.mean(performances[i,j*int(iterations/a):(j+1)*int(iterations/a)])
            general_perf2[i,j]=np.mean(performances2[i,j*int(iterations/a):(j+1)*int(iterations/a)])
    return general_perf,general_perf2