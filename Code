import numpy as np
import matplotlib.pyplot as plt
#1) task
N = 1000
Nr = 100

def kick_out(epochs):
    all_F_plus, all_e_minus_and_f_plus = 0, 0
    
    for i in range(epochs):
        a = np.all(np.random.choice(2, 1, p = [0.3, 0.7]))
        d = np.all(np.random.choice(2, 1, p = [0.8, 0.2] if a else [0.3, 0.7]))
        f = np.all(np.random.choice(2, 1, p = [0.3, 0.7] if d else [0.8, 0.2]))
        if not f: continue
        b = np.all(np.random.choice(2, 1, p = [0.2, 0.8] if a else [0.7, 0.3]))
        
        e = np.all(np.random.choice(2, 1, p = [0.8, 0.2] if (a and b) else [0.3, 0.7]\
        if (a and not b) else [0.5, 0.5] if (not a and b) else [0.6, 0.4]))
        
        all_F_plus += 1
        if not e: all_e_minus_and_f_plus +=1                    
                            
    return all_e_minus_and_f_plus / all_F_plus

def analyze_method(number_of_realizations, method, epochs):
    result_array = np.zeros((1, number_of_realizations))
    
    for i in range(number_of_realizations):
        result_array[0][i] = method(epochs)
    
    if method.__name__ == "Gibbs_method":
        method_mean = np.mean(result_array[0][5:])
        method_std = np.std(result_array[0][5:])
    else:
        method_mean = np.mean(result_array)
        method_std = np.std(result_array)
        
    #plt.figure()
    plt.hist(result_array.T)
    plt.title(f"Histogram for {number_of_realizations} realizations, {method.__name__}")
    plt.text(0.6, 15, f"mean = {method_mean:.3f}, std = {method_std:.3f}, accurate = 0.644")
    plt.show()
    
#analyze_method(Nr, kick_out, N)

def weighting_method(epochs):
    all_e_minus_d_minus, all_e_plus_d_minus = 0, 0
    all_e_minus_d_plus, all_e_plus_d_plus = 0, 0
    
    for i in range(epochs):
        a = np.all(np.random.choice(2, 1, p = [0.3, 0.7]))
        d = np.all(np.random.choice(2, 1, p = [0.8, 0.2] if a else [0.3, 0.7]))
        b = np.all(np.random.choice(2, 1, p = [0.2, 0.8] if a else [0.7, 0.3]))
        e = np.all(np.random.choice(2, 1, p = [0.8, 0.2] if (a and b) else [0.3, 0.7]\
        if (a and not b) else [0.5, 0.5] if (not a and b) else [0.6, 0.4]))
            
        if not e and not d: all_e_minus_d_minus += 1
        if e and not d: all_e_plus_d_minus += 1
        if not e and d: all_e_minus_d_plus += 1
        if e and d: all_e_plus_d_plus += 1
        
    p_e_minus_f_plus = all_e_minus_d_minus*0.2 + 0.7*all_e_minus_d_plus
    p_e_plus_f_plus = all_e_plus_d_minus*0.2 + 0.7*all_e_plus_d_plus
        
    p_e_minus_f_plus_final = p_e_minus_f_plus/(p_e_minus_f_plus + p_e_plus_f_plus)
    
    return p_e_minus_f_plus_final
    
#analyze_method(Nr, weighting_method, N)



    
def Gibbs_method(epochs):
    def p_b_of_a(a,b):
        if a and b: return 0.8
        elif a and not b: return 0.2
        elif not a and b: return 0.3
        else: return 0.7

    def p_d_of_a(a,d):
        if a and d: return 0.2
        elif a and not d: return 0.8
        elif not a and d: return 0.7
        else: return 0.3
    
    def p_f_of_d(d):
        if d: return 0.7
        else: return 0.2
        
    # def p_a(a):
    #     if a: return 0.7
    #     else: return 0.3
    p_a = 0.7
    
    def p_e_of_ab(a, b, e):
        if a and b and e: return 0.2
        elif a and not b and e: return 0.7
        elif not a and b and e: return 0.5
        elif not a and not b and e: return 0.4
        elif a and b and not e: return 0.8
        elif a and not b and not e: return 0.3
        elif not a and b and not e: return 0.5
        elif not a and not b and not e: return 0.6
    
    def p_a_of_ebd(a,e,b,d):
        return p_a*p_e_of_ab(a, b, e)*p_b_of_a(a, b)*p_d_of_a(a, d)
        #return p_a(a)*p_e_of_ab(a, b, e)*p_b_of_a(a, b)*p_d_of_a(a, d)
    def p_b_of_aed(a,b,e):
        return p_e_of_ab(a, b,e)*p_b_of_a(a,b)
    def p_d_of_abe(a,d):
        return p_f_of_d(d)*p_d_of_a(a,d)
    def p_e_of_abd(a,b,e):
        return p_e_of_ab(a, b, e)
    
    init = np.random.choice(2, 4)
    a, b, d, e = np.all(init[0]), np.all(init[1]), np.all(init[2]), np.all(init[3])
    all_e_minus = 0
    for i in range(epochs):
        P_a_plus = p_a_of_ebd(True, b, d, e)
        P_a_minus = p_a_of_ebd(False, b, d, e)
        P_a_plus = P_a_plus / (P_a_plus + P_a_minus)
        #print(P_a_plus)
        a = np.all(np.random.choice(2, 1, p = [1-P_a_plus, P_a_plus]))
        
        P_b_plus = p_b_of_aed(a, True, e)
        P_b_minus = p_b_of_aed(a, False, e)
        P_b_plus = P_b_plus/(P_b_plus + P_b_minus)
        b = np.all(np.random.choice(2, 1, p = [1-P_b_plus, P_b_plus]))
        
        P_e_plus = p_e_of_abd(a, b, True)
        P_e_minus = p_e_of_abd(a, b, False)
        P_e_plus = P_e_plus/(P_e_plus + P_e_minus)
        e = np.all(np.random.choice(2, 1, p = [1-P_e_plus, P_e_plus]))
        
        P_d_plus = p_d_of_abe(a, True)*p_f_of_d(a)
        P_d_minus = p_d_of_abe(a, False)*p_f_of_d(a)
        P_d_plus = P_d_plus/(P_d_plus + P_d_minus)
        d = np.all(np.random.choice(2, 1, p = [1-P_d_plus, P_d_plus]))
              
        if not e: all_e_minus += 1
    
    return all_e_minus / epochs

#analyze_method(Nr, Gibbs_method, N)




#2. task
def simulate_trajectory(time_interval):
    trajectory = np.zeros(time_interval)
    velocity = np.zeros(time_interval)
    mi = np.random.uniform(-2, 2)
    x0,v0 = 0, np.random.normal()
    trajectory[0], velocity[0] = x0, v0
   
    for i in range(time_interval-1):
        #motion model
        velocity[i+1] = velocity[i] + 0.2*(mi-velocity[i]) + 0.32*np.random.normal()
        
        #position model
        trajectory[i+1] = trajectory[i] + velocity[i]
        
    return trajectory, velocity, mi

def analyze_trajectory(trajectory, velocity, mi, special = "none"):
    plt.plot(trajectory)
    if special == "none":
        plt.title("simuladed trajectory - position[time]")
    else:
        plt.title("predicted trajectory - position[time]")
    plt.xlabel("time[s]")
    plt.ylabel("position")
    plt.text(20, mi*15, f"mi = {mi:.3f}")
    plt.show()
    
    plt.figure()
    plt.plot(velocity)
    if special == "none":
        plt.title("simuladed trajectory - velocity[time]")
    else:
        plt.title("predicted trajectory - velocity[time]")
    plt.xlabel("time[s]")
    plt.ylabel("velocity")
    plt.text(20, mi+0.04, f"mi = {mi:.3f}")
    plt.plot([0, len(trajectory)], [mi, mi], "--")
    plt.show()
    
s = simulate_trajectory(50)
real_x, velocity, mi = s[0], s[1], s[2]
analyze_trajectory(s[0], s[1], s[2])


#measuring model    
def trajectory_and_noise(trajectory):
    e = np.zeros(len(trajectory))
    for i in range(len(trajectory)):
        e[i] = (-1)**np.random.choice(2, 1, p = [0.95, 0.05])*trajectory[i]\
            + np.random.laplace(0, np.sqrt(np.abs(trajectory[i]))/5)
        
    return e
    
def plot_noisy_trajectory(trajectory, mi):
    plt.plot(trajectory)
    plt.title("noisy trajectory")
    plt.xlabel("time[s]")
    plt.ylabel("position")
    plt.text(20, mi*15, f"mi = {mi:.3f}")
    plt.show()
    

def plot_trajectory(trajectory, mi):
    plt.plot(trajectory)
    plt.title("trajectory")
    plt.xlabel("time[s]")
    plt.ylabel("position")
    plt.text(20, mi*15, f"mi = {mi:.3f}")
    plt.show()
    
e_x = trajectory_and_noise(s[0])
plot_noisy_trajectory(e_x, mi)

#particle filter
def simulate_estimated_trajectory(time_interval, mi):
    trajectory, velocity = np.zeros(time_interval), np.zeros(time_interval)
    for i in range(1, time_interval-1):
        #motion model
        velocity[i+1] = velocity[i] + 0.2*(mi-velocity[i])
        
        #position model
        trajectory[i+1] = trajectory[i] + velocity[i]

    return trajectory, velocity, mi


#estimation = simulate_estimated_trajectory(50, mi)
#analyze_trajectory(estimation[0], estimation[1], estimation[2], "lol")


def plot_estimated_real_noisy_trajectories(real_x, real_v, est_x, est_v, e_x, mi):
    plt.plot(real_x)
    plt.plot(est_x)
    plt.plot(e_x)
    plt.title("All trajectories - position[time]")
    plt.xlabel("time[s]")
    plt.ylabel("position")
    plt.legend(["real", "estimated", "measured"])
    plt.text(20, mi*15, f"mi = {mi:.3f}")
    plt.show()
    
    plt.figure()
    plt.plot(real_v)
    plt.plot(est_v)
    plt.title("All trajectories - velocity[time]")
    plt.xlabel("time[s]")
    plt.ylabel("velocity")
    plt.legend(["real", "estimated"])
    plt.text(20, mi+0.04, f"mi = {mi:.3f}")
    plt.plot([0, len(real_x)], [mi, mi], "--")
    plt.show()

#plot_estimated_real_noisy_trajectories(real_x, velocity, estimation[0], estimation[1],e_x, mi)    
    
    
def plot_particles(trajectory_pred, weights, e_x, real_x, t):
    plt.plot([real_x[t], real_x[t]], [0, np.max(weights[:, t])], "--")
    plt.plot([e_x[t], e_x[t]], [0, np.max(weights[:, t])], "--")
    plt.scatter(trajectory_pred[:, t], weights[:, t])
    plt.xlabel("prediction")
    plt.ylabel("weight_of_prediction")
    plt.title("before resampling")
    plt.legend(["real_x", "noisy_x", "weights_of_prediction"])
    plt.show()    
    
    
def particle_filter(number_of_particles,time_interval, e_x, real_x, real_v, mi):
    mi_of_particles = np.random.uniform(-2, 2, number_of_particles)
    position = np.zeros((number_of_particles, time_interval))
    velocity = np.zeros((number_of_particles, time_interval))
    weights = np.zeros((number_of_particles, time_interval))
    estimation_t_x = np.zeros(time_interval)
    estimation_t_v = np.zeros(time_interval)
    trajectory_pred = np.zeros((number_of_particles, time_interval))
    velocity_pred = np.zeros((number_of_particles, time_interval))
    
    for j in range(number_of_particles):
        velocity[j][0] = np.random.normal()
        weights[j][0] = 1/number_of_particles
    
    for t in range(1, time_interval):
        for j in range(number_of_particles):
            #prediction
            velocity_pred[j][t] = velocity[j][t-1] + 0.2*(mi_of_particles[j]-velocity[j][t-1])
            trajectory_pred[j][t] = position[j][t-1] + velocity_pred[j][t]
            
            
            #weightening
            weights[j, t] = weights[j][t-1]*np.exp(-((e_x[t] - trajectory_pred[j][t])**2/2))
        
   
        weights_sum = np.sum(weights[:, t])
        for j in range(number_of_particles):
            weights[j, t] = weights[j, t] / weights_sum
            #estimation
            estimation_t_x[t] += weights[j, t]*trajectory_pred[j][t]
            estimation_t_v[t] += weights[j, t]*velocity_pred[j][t]
    
        print(estimation_t_x[t])
        
        #no resampling
        #position[:, t] = trajectory_pred[:, t]
        #velocity[:, t] = velocity_pred[:, t]
        
        # #always resampling
        # position[:, t] = np.random.choice(trajectory_pred[:, t], number_of_particles, p = list(weights[:, t]))
        # velocity[:, t] = np.random.choice(velocity_pred[:, t], number_of_particles, p = list(weights[:, t]))
        # for j in range(number_of_particles):
        #     weights[j, t] = 1/number_of_particles
        
        #conditional resampling
        if 1/(np.sum(weights[:, t]*weights[:, t])) > number_of_particles/2:
            position[:, t] = trajectory_pred[:, t]
            velocity[:, t] = velocity_pred[:, t]
        else:
            
            position[:, t] = np.random.choice(trajectory_pred[:, t], number_of_particles, p = list(weights[:, t]))
            velocity[:, t] = np.random.choice(velocity_pred[:, t], number_of_particles, p = list(weights[:, t]))
            for j in range(number_of_particles):
                weights[j, t] = 1/number_of_particles
        

    #grafs
    
    # plot_estimated_real_noisy_trajectories(real_x, real_v, estimation_t_x, estimation_t_v, e_x, mi)
    
    # plot_particles(trajectory_pred, weights, e_x, real_x, 5)
    # plot_particles(trajectory_pred, weights, e_x, real_x, 15)
    # plot_particles(trajectory_pred, weights, e_x, real_x, 25)    
    # plot_particles(trajectory_pred, weights, e_x, real_x, 35)
    # plot_particles(trajectory_pred, weights, e_x, real_x, 40)
    # plot_particles(trajectory_pred, weights, e_x, real_x, 45)
    
    
    #average error
    average_error = (estimation_t_x-real_x)*(estimation_t_x-real_x)
                          
    return average_error
    
p_f = particle_filter(20, 50, e_x, real_x, velocity, mi)
print(p_f)

#plot_trajectory(p_f, 0)


#monte Karlo
error = np.zeros((100, 50))
for i in range(100):
    error[i, :] = particle_filter(20,50, e_x, real_x, velocity, mi)

average_errors = np.sqrt(np.mean(error, 0))

plt.plot(average_errors)
plt.title("conditional resampling")
plt.xlabel("Time[s]")
plt.ylabel("Average errors[time]")
plt.show()
    
    
    
   
