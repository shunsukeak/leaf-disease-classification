# This is for plotting graph 
import matplotlib.pyplot as plt

def calculate_total_accuracy(binary_accuracy, P_disease, P_non, A_multi):
    return binary_accuracy * P_disease * A_multi + binary_accuracy * P_non

binary_accuracy = 0.9971

# propotion of dataset
disease_count = 39211
healthy_count = 15084
P_disease = disease_count / (disease_count + healthy_count)
P_non = healthy_count / (disease_count + healthy_count)

A_multi_values = [0.9475, 0.9513, 0.9424, 0.9353, 0.9283, 0.9040, 0.8771, 0.8271, 0.7394, 0.6504]

# calcurate accuracy for proposed method
total_accuracies = {}
for A_multi in A_multi_values:
    A_total = calculate_total_accuracy(binary_accuracy, P_disease, P_non, A_multi)
    total_accuracies[A_multi] = A_total

# create graphs
plt.figure(figsize=(8, 6))

x_values = [78.43, 70.27, 63.54, 56.94, 50.66, 43.99, 37.28, 29.93, 22.73, 19.00]  
y_values = list(total_accuracies.values()) 
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='proposed')


# new_x_values = [109.77, 90.55, 49.74, 34.33, 27.15, 21.85, 17.66]  
# new_y_values = [0.9193, 0.9105, 0.9033, 0.8297, 0.7516, 0.5749, 0.3713]  

new_x_values = [90.55, 49.74, 34.33, 27.15, 21.85, 17.66]  
new_y_values = [0.9105, 0.9033, 0.8297, 0.7516, 0.5749, 0.3713] 
plt.plot(new_x_values, new_y_values, marker='s', linestyle='--', color='r', label='baseline')

plt.xlabel('Data size [MB]')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('./accuracy_comparison.pdf')
plt.show()
plt.close()


