\ Model VRP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  1640 x[0,1,0] + 1640 x[0,1,1] + 1640 x[0,1,2] + 2208 x[0,2,0]
   + 2208 x[0,2,1] + 2208 x[0,2,2] + 2472 x[0,3,0] + 2472 x[0,3,1]
   + 2472 x[0,3,2] + 3300 x[1,4,0] + 3300 x[1,4,1] + 3300 x[1,4,2]
   + 3195 x[1,5,0] + 3195 x[1,5,1] + 3195 x[1,5,2] + 2843 x[1,6,0]
   + 2843 x[1,6,1] + 2843 x[1,6,2] + 2558 x[1,7,0] + 2558 x[1,7,1]
   + 2558 x[1,7,2] + 1758 x[1,8,0] + 1758 x[1,8,1] + 1758 x[1,8,2]
   + 3077 x[2,4,0] + 3077 x[2,4,1] + 3077 x[2,4,2] + 2947 x[2,5,0]
   + 2947 x[2,5,1] + 2947 x[2,5,2] + 2656 x[2,6,0] + 2656 x[2,6,1]
   + 2656 x[2,6,2] + 2755 x[2,7,0] + 2755 x[2,7,1] + 2755 x[2,7,2]
   + 1119 x[2,8,0] + 1119 x[2,8,1] + 1119 x[2,8,2] + 3064 x[3,4,0]
   + 3064 x[3,4,1] + 3064 x[3,4,2] + 2924 x[3,5,0] + 2924 x[3,5,1]
   + 2924 x[3,5,2] + 2669 x[3,6,0] + 2669 x[3,6,1] + 2669 x[3,6,2]
   + 2936 x[3,7,0] + 2936 x[3,7,1] + 2936 x[3,7,2] + 866 x[3,8,0]
   + 866 x[3,8,1] + 866 x[3,8,2] + 168 x[4,5,0] + 168 x[4,5,1]
   + 168 x[4,5,2] + 472 x[4,6,0] + 472 x[4,6,1] + 472 x[4,6,2]
   + 1898 x[4,7,0] + 1898 x[4,7,1] + 1898 x[4,7,2] + 2568 x[4,8,0]
   + 2568 x[4,8,1] + 2568 x[4,8,2] + 168 x[5,4,0] + 168 x[5,4,1]
   + 168 x[5,4,2] + 424 x[5,6,0] + 424 x[5,6,1] + 424 x[5,6,2]
   + 1936 x[5,7,0] + 1936 x[5,7,1] + 1936 x[5,7,2] + 2409 x[5,8,0]
   + 2409 x[5,8,1] + 2409 x[5,8,2] + 472 x[6,4,0] + 472 x[6,4,1]
   + 472 x[6,4,2] + 424 x[6,5,0] + 424 x[6,5,1] + 424 x[6,5,2]
   + 1548 x[6,7,0] + 1548 x[6,7,1] + 1548 x[6,7,2] + 2261 x[6,8,0]
   + 2261 x[6,8,1] + 2261 x[6,8,2] + 1898 x[7,4,0] + 1898 x[7,4,1]
   + 1898 x[7,4,2] + 1936 x[7,5,0] + 1936 x[7,5,1] + 1936 x[7,5,2]
   + 1548 x[7,6,0] + 1548 x[7,6,1] + 1548 x[7,6,2] + 2987 x[7,8,0]
   + 2987 x[7,8,1] + 2987 x[7,8,2] + 2568 x[8,4,0] + 2568 x[8,4,1]
   + 2568 x[8,4,2] + 2409 x[8,5,0] + 2409 x[8,5,1] + 2409 x[8,5,2]
   + 2261 x[8,6,0] + 2261 x[8,6,1] + 2261 x[8,6,2] + 2987 x[8,7,0]
   + 2987 x[8,7,1] + 2987 x[8,7,2] + 4822 x[4,0,0] + 4822 x[4,0,1]
   + 4822 x[4,0,2] + 4737 x[5,0,0] + 4737 x[5,0,1] + 4737 x[5,0,2]
   + 4351 x[6,0,0] + 4351 x[6,0,1] + 4351 x[6,0,2] + 3634 x[7,0,0]
   + 3634 x[7,0,1] + 3634 x[7,0,2] + 3326 x[8,0,0] + 3326 x[8,0,1]
   + 3326 x[8,0,2] + 1e+12 tau[1,0] + 1e+12 tau[2,0] + 1e+12 tau[3,0]
   + 1e+12 tau[4,0] + 1e+12 tau[5,0] + 1e+12 tau[6,0] + 1e+12 tau[7,0]
   + 1e+12 tau[8,0] + 1e+12 tau[1,1] + 1e+12 tau[2,1] + 1e+12 tau[3,1]
   + 1e+12 tau[4,1] + 1e+12 tau[5,1] + 1e+12 tau[6,1] + 1e+12 tau[7,1]
   + 1e+12 tau[8,1] + 1e+12 tau[1,2] + 1e+12 tau[2,2] + 1e+12 tau[3,2]
   + 1e+12 tau[4,2] + 1e+12 tau[5,2] + 1e+12 tau[6,2] + 1e+12 tau[7,2]
   + 1e+12 tau[8,2] - 1e+06 launch[0] - 1e+06 launch[1] - 1e+06 launch[2]
   + 1e+06 land[0] + 1e+06 land[1] + 1e+06 land[2] - 1.68e+16 Constant
Subject To
 Launch_site[0]: x[0,1,0] + x[0,2,0] + x[0,3,0] - x_k[0] = 0
 Launch_site[1]: x[0,1,1] + x[0,2,1] + x[0,3,1] - x_k[1] = 0
 Launch_site[2]: x[0,1,2] + x[0,2,2] + x[0,3,2] - x_k[2] = 0
 Landing_site[0]: x[4,0,0] + x[5,0,0] + x[6,0,0] + x[7,0,0] + x[8,0,0]
   - x_k[0] = 0
 Landing_site[1]: x[4,0,1] + x[5,0,1] + x[6,0,1] + x[7,0,1] + x[8,0,1]
   - x_k[1] = 0
 Landing_site[2]: x[4,0,2] + x[5,0,2] + x[6,0,2] + x[7,0,2] + x[8,0,2]
   - x_k[2] = 0
 Visit_pizzeria[1]: x[0,1,0] + x[0,1,1] + x[0,1,2] = 1
 Visit_pizzeria[2]: x[0,2,0] + x[0,2,1] + x[0,2,2] = 1
 Visit_pizzeria[3]: x[0,3,0] + x[0,3,1] + x[0,3,2] = 1
 Visit_customer[4]: x[1,4,0] + x[1,4,1] + x[1,4,2] + x[2,4,0] + x[2,4,1]
   + x[2,4,2] + x[3,4,0] + x[3,4,1] + x[3,4,2] + x[5,4,0] + x[5,4,1]
   + x[5,4,2] + x[6,4,0] + x[6,4,1] + x[6,4,2] + x[7,4,0] + x[7,4,1]
   + x[7,4,2] + x[8,4,0] + x[8,4,1] + x[8,4,2] = 1
 Visit_customer[5]: x[1,5,0] + x[1,5,1] + x[1,5,2] + x[2,5,0] + x[2,5,1]
   + x[2,5,2] + x[3,5,0] + x[3,5,1] + x[3,5,2] + x[4,5,0] + x[4,5,1]
   + x[4,5,2] + x[6,5,0] + x[6,5,1] + x[6,5,2] + x[7,5,0] + x[7,5,1]
   + x[7,5,2] + x[8,5,0] + x[8,5,1] + x[8,5,2] = 1
 Visit_customer[6]: x[1,6,0] + x[1,6,1] + x[1,6,2] + x[2,6,0] + x[2,6,1]
   + x[2,6,2] + x[3,6,0] + x[3,6,1] + x[3,6,2] + x[4,6,0] + x[4,6,1]
   + x[4,6,2] + x[5,6,0] + x[5,6,1] + x[5,6,2] + x[7,6,0] + x[7,6,1]
   + x[7,6,2] + x[8,6,0] + x[8,6,1] + x[8,6,2] = 1
 Visit_customer[7]: x[1,7,0] + x[1,7,1] + x[1,7,2] + x[2,7,0] + x[2,7,1]
   + x[2,7,2] + x[3,7,0] + x[3,7,1] + x[3,7,2] + x[4,7,0] + x[4,7,1]
   + x[4,7,2] + x[5,7,0] + x[5,7,1] + x[5,7,2] + x[6,7,0] + x[6,7,1]
   + x[6,7,2] + x[8,7,0] + x[8,7,1] + x[8,7,2] = 1
 Visit_customer[8]: x[1,8,0] + x[1,8,1] + x[1,8,2] + x[2,8,0] + x[2,8,1]
   + x[2,8,2] + x[3,8,0] + x[3,8,1] + x[3,8,2] + x[4,8,0] + x[4,8,1]
   + x[4,8,2] + x[5,8,0] + x[5,8,1] + x[5,8,2] + x[6,8,0] + x[6,8,1]
   + x[6,8,2] + x[7,8,0] + x[7,8,1] + x[7,8,2] = 1
 Leave_customer[4,0]: x[1,4,0] + x[2,4,0] + x[3,4,0] - x[4,5,0] - x[4,6,0]
   - x[4,7,0] - x[4,8,0] + x[5,4,0] + x[6,4,0] + x[7,4,0] + x[8,4,0]
   - x[4,0,0] = 0
 Leave_customer[4,1]: x[1,4,1] + x[2,4,1] + x[3,4,1] - x[4,5,1] - x[4,6,1]
   - x[4,7,1] - x[4,8,1] + x[5,4,1] + x[6,4,1] + x[7,4,1] + x[8,4,1]
   - x[4,0,1] = 0
 Leave_customer[4,2]: x[1,4,2] + x[2,4,2] + x[3,4,2] - x[4,5,2] - x[4,6,2]
   - x[4,7,2] - x[4,8,2] + x[5,4,2] + x[6,4,2] + x[7,4,2] + x[8,4,2]
   - x[4,0,2] = 0
 Leave_customer[5,0]: x[1,5,0] + x[2,5,0] + x[3,5,0] + x[4,5,0] - x[5,4,0]
   - x[5,6,0] - x[5,7,0] - x[5,8,0] + x[6,5,0] + x[7,5,0] + x[8,5,0]
   - x[5,0,0] = 0
 Leave_customer[5,1]: x[1,5,1] + x[2,5,1] + x[3,5,1] + x[4,5,1] - x[5,4,1]
   - x[5,6,1] - x[5,7,1] - x[5,8,1] + x[6,5,1] + x[7,5,1] + x[8,5,1]
   - x[5,0,1] = 0
 Leave_customer[5,2]: x[1,5,2] + x[2,5,2] + x[3,5,2] + x[4,5,2] - x[5,4,2]
   - x[5,6,2] - x[5,7,2] - x[5,8,2] + x[6,5,2] + x[7,5,2] + x[8,5,2]
   - x[5,0,2] = 0
 Leave_customer[6,0]: x[1,6,0] + x[2,6,0] + x[3,6,0] + x[4,6,0] + x[5,6,0]
   - x[6,4,0] - x[6,5,0] - x[6,7,0] - x[6,8,0] + x[7,6,0] + x[8,6,0]
   - x[6,0,0] = 0
 Leave_customer[6,1]: x[1,6,1] + x[2,6,1] + x[3,6,1] + x[4,6,1] + x[5,6,1]
   - x[6,4,1] - x[6,5,1] - x[6,7,1] - x[6,8,1] + x[7,6,1] + x[8,6,1]
   - x[6,0,1] = 0
 Leave_customer[6,2]: x[1,6,2] + x[2,6,2] + x[3,6,2] + x[4,6,2] + x[5,6,2]
   - x[6,4,2] - x[6,5,2] - x[6,7,2] - x[6,8,2] + x[7,6,2] + x[8,6,2]
   - x[6,0,2] = 0
 Leave_customer[7,0]: x[1,7,0] + x[2,7,0] + x[3,7,0] + x[4,7,0] + x[5,7,0]
   + x[6,7,0] - x[7,4,0] - x[7,5,0] - x[7,6,0] - x[7,8,0] + x[8,7,0]
   - x[7,0,0] = 0
 Leave_customer[7,1]: x[1,7,1] + x[2,7,1] + x[3,7,1] + x[4,7,1] + x[5,7,1]
   + x[6,7,1] - x[7,4,1] - x[7,5,1] - x[7,6,1] - x[7,8,1] + x[8,7,1]
   - x[7,0,1] = 0
 Leave_customer[7,2]: x[1,7,2] + x[2,7,2] + x[3,7,2] + x[4,7,2] + x[5,7,2]
   + x[6,7,2] - x[7,4,2] - x[7,5,2] - x[7,6,2] - x[7,8,2] + x[8,7,2]
   - x[7,0,2] = 0
 Leave_customer[8,0]: x[1,8,0] + x[2,8,0] + x[3,8,0] + x[4,8,0] + x[5,8,0]
   + x[6,8,0] + x[7,8,0] - x[8,4,0] - x[8,5,0] - x[8,6,0] - x[8,7,0]
   - x[8,0,0] = 0
 Leave_customer[8,1]: x[1,8,1] + x[2,8,1] + x[3,8,1] + x[4,8,1] + x[5,8,1]
   + x[6,8,1] + x[7,8,1] - x[8,4,1] - x[8,5,1] - x[8,6,1] - x[8,7,1]
   - x[8,0,1] = 0
 Leave_customer[8,2]: x[1,8,2] + x[2,8,2] + x[3,8,2] + x[4,8,2] + x[5,8,2]
   + x[6,8,2] + x[7,8,2] - x[8,4,2] - x[8,5,2] - x[8,6,2] - x[8,7,2]
   - x[8,0,2] = 0
 Leave_pizzeria[1,0]: - x[0,1,0] + x[1,4,0] + x[1,5,0] + x[1,6,0]
   + x[1,7,0] + x[1,8,0] = 0
 Leave_pizzeria[1,1]: - x[0,1,1] + x[1,4,1] + x[1,5,1] + x[1,6,1]
   + x[1,7,1] + x[1,8,1] = 0
 Leave_pizzeria[1,2]: - x[0,1,2] + x[1,4,2] + x[1,5,2] + x[1,6,2]
   + x[1,7,2] + x[1,8,2] = 0
 Leave_pizzeria[2,0]: - x[0,2,0] + x[2,4,0] + x[2,5,0] + x[2,6,0]
   + x[2,7,0] + x[2,8,0] = 0
 Leave_pizzeria[2,1]: - x[0,2,1] + x[2,4,1] + x[2,5,1] + x[2,6,1]
   + x[2,7,1] + x[2,8,1] = 0
 Leave_pizzeria[2,2]: - x[0,2,2] + x[2,4,2] + x[2,5,2] + x[2,6,2]
   + x[2,7,2] + x[2,8,2] = 0
 Leave_pizzeria[3,0]: - x[0,3,0] + x[3,4,0] + x[3,5,0] + x[3,6,0]
   + x[3,7,0] + x[3,8,0] = 0
 Leave_pizzeria[3,1]: - x[0,3,1] + x[3,4,1] + x[3,5,1] + x[3,6,1]
   + x[3,7,1] + x[3,8,1] = 0
 Leave_pizzeria[3,2]: - x[0,3,2] + x[3,4,2] + x[3,5,2] + x[3,6,2]
   + x[3,7,2] + x[3,8,2] = 0
 time_bound_on_pizzeria[1,0]: 1e+06 x[0,1,0] - tau[1,0] <= 999400
 time_bound_on_pizzeria[1,1]: 1e+06 x[0,1,1] - tau[1,1] <= 999400
 time_bound_on_pizzeria[1,2]: 1e+06 x[0,1,2] - tau[1,2] <= 999400
 time_bound_on_pizzeria[2,0]: 1e+06 x[0,2,0] - tau[2,0] <= 999300
 time_bound_on_pizzeria[2,1]: 1e+06 x[0,2,1] - tau[2,1] <= 999300
 time_bound_on_pizzeria[2,2]: 1e+06 x[0,2,2] - tau[2,2] <= 999300
 time_bound_on_pizzeria[3,0]: 1e+06 x[0,3,0] - tau[3,0] <= 999200
 time_bound_on_pizzeria[3,1]: 1e+06 x[0,3,1] - tau[3,1] <= 999200
 time_bound_on_pizzeria[3,2]: 1e+06 x[0,3,2] - tau[3,2] <= 999200
 Time_window_pizzeria[1,0]: 1e+06 x[0,1,0] - tau[1,0] <= 999918
 Time_window_pizzeria[1,1]: 1e+06 x[0,1,1] - tau[1,1] <= 999918
 Time_window_pizzeria[1,2]: 1e+06 x[0,1,2] - tau[1,2] <= 999918
 Time_window_pizzeria[2,0]: 1e+06 x[0,2,0] - tau[2,0] <= 999889.6
 Time_window_pizzeria[2,1]: 1e+06 x[0,2,1] - tau[2,1] <= 999889.6
 Time_window_pizzeria[2,2]: 1e+06 x[0,2,2] - tau[2,2] <= 999889.6
 Time_window_pizzeria[3,0]: 1e+06 x[0,3,0] - tau[3,0] <= 999876.4
 Time_window_pizzeria[3,1]: 1e+06 x[0,3,1] - tau[3,1] <= 999876.4
 Time_window_pizzeria[3,2]: 1e+06 x[0,3,2] - tau[3,2] <= 999876.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,4,0]:
   1e+06 x[1,4,0] + tau[1,0] - tau[4,0] <= 999835
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,4,1]:
   1e+06 x[1,4,1] + tau[1,1] - tau[4,1] <= 999835
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,4,2]:
   1e+06 x[1,4,2] + tau[1,2] - tau[4,2] <= 999835
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,5,0]:
   1e+06 x[1,5,0] + tau[1,0] - tau[5,0] <= 999840.25
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,5,1]:
   1e+06 x[1,5,1] + tau[1,1] - tau[5,1] <= 999840.25
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,5,2]:
   1e+06 x[1,5,2] + tau[1,2] - tau[5,2] <= 999840.25
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,6,0]:
   1e+06 x[1,6,0] + tau[1,0] - tau[6,0] <= 999857.85
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,6,1]:
   1e+06 x[1,6,1] + tau[1,1] - tau[6,1] <= 999857.85
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,6,2]:
   1e+06 x[1,6,2] + tau[1,2] - tau[6,2] <= 999857.85
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,7,0]:
   1e+06 x[1,7,0] + tau[1,0] - tau[7,0] <= 999872.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,7,1]:
   1e+06 x[1,7,1] + tau[1,1] - tau[7,1] <= 999872.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,7,2]:
   1e+06 x[1,7,2] + tau[1,2] - tau[7,2] <= 999872.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,8,0]:
   1e+06 x[1,8,0] + tau[1,0] - tau[8,0] <= 999912.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,8,1]:
   1e+06 x[1,8,1] + tau[1,1] - tau[8,1] <= 999912.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,8,2]:
   1e+06 x[1,8,2] + tau[1,2] - tau[8,2] <= 999912.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,4,0]:
   1e+06 x[2,4,0] + tau[2,0] - tau[4,0] <= 999846.15
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,4,1]:
   1e+06 x[2,4,1] + tau[2,1] - tau[4,1] <= 999846.15
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,4,2]:
   1e+06 x[2,4,2] + tau[2,2] - tau[4,2] <= 999846.15
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,5,0]:
   1e+06 x[2,5,0] + tau[2,0] - tau[5,0] <= 999852.65
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,5,1]:
   1e+06 x[2,5,1] + tau[2,1] - tau[5,1] <= 999852.65
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,5,2]:
   1e+06 x[2,5,2] + tau[2,2] - tau[5,2] <= 999852.65
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,6,0]:
   1e+06 x[2,6,0] + tau[2,0] - tau[6,0] <= 999867.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,6,1]:
   1e+06 x[2,6,1] + tau[2,1] - tau[6,1] <= 999867.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,6,2]:
   1e+06 x[2,6,2] + tau[2,2] - tau[6,2] <= 999867.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,7,0]:
   1e+06 x[2,7,0] + tau[2,0] - tau[7,0] <= 999862.25
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,7,1]:
   1e+06 x[2,7,1] + tau[2,1] - tau[7,1] <= 999862.25
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,7,2]:
   1e+06 x[2,7,2] + tau[2,2] - tau[7,2] <= 999862.25
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,8,0]:
   1e+06 x[2,8,0] + tau[2,0] - tau[8,0] <= 999944.05
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,8,1]:
   1e+06 x[2,8,1] + tau[2,1] - tau[8,1] <= 999944.05
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,8,2]:
   1e+06 x[2,8,2] + tau[2,2] - tau[8,2] <= 999944.05
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,4,0]:
   1e+06 x[3,4,0] + tau[3,0] - tau[4,0] <= 999846.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,4,1]:
   1e+06 x[3,4,1] + tau[3,1] - tau[4,1] <= 999846.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,4,2]:
   1e+06 x[3,4,2] + tau[3,2] - tau[4,2] <= 999846.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,5,0]:
   1e+06 x[3,5,0] + tau[3,0] - tau[5,0] <= 999853.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,5,1]:
   1e+06 x[3,5,1] + tau[3,1] - tau[5,1] <= 999853.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,5,2]:
   1e+06 x[3,5,2] + tau[3,2] - tau[5,2] <= 999853.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,6,0]:
   1e+06 x[3,6,0] + tau[3,0] - tau[6,0] <= 999866.55
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,6,1]:
   1e+06 x[3,6,1] + tau[3,1] - tau[6,1] <= 999866.55
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,6,2]:
   1e+06 x[3,6,2] + tau[3,2] - tau[6,2] <= 999866.55
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,7,0]:
   1e+06 x[3,7,0] + tau[3,0] - tau[7,0] <= 999853.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,7,1]:
   1e+06 x[3,7,1] + tau[3,1] - tau[7,1] <= 999853.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,7,2]:
   1e+06 x[3,7,2] + tau[3,2] - tau[7,2] <= 999853.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,8,0]:
   1e+06 x[3,8,0] + tau[3,0] - tau[8,0] <= 999956.7
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,8,1]:
   1e+06 x[3,8,1] + tau[3,1] - tau[8,1] <= 999956.7
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,8,2]:
   1e+06 x[3,8,2] + tau[3,2] - tau[8,2] <= 999956.7
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,5,0]:
   1e+06 x[4,5,0] + tau[4,0] - tau[5,0] <= 999991.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,5,1]:
   1e+06 x[4,5,1] + tau[4,1] - tau[5,1] <= 999991.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,5,2]:
   1e+06 x[4,5,2] + tau[4,2] - tau[5,2] <= 999991.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,6,0]:
   1e+06 x[4,6,0] + tau[4,0] - tau[6,0] <= 999976.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,6,1]:
   1e+06 x[4,6,1] + tau[4,1] - tau[6,1] <= 999976.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,6,2]:
   1e+06 x[4,6,2] + tau[4,2] - tau[6,2] <= 999976.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,7,0]:
   1e+06 x[4,7,0] + tau[4,0] - tau[7,0] <= 999905.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,7,1]:
   1e+06 x[4,7,1] + tau[4,1] - tau[7,1] <= 999905.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,7,2]:
   1e+06 x[4,7,2] + tau[4,2] - tau[7,2] <= 999905.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,8,0]:
   1e+06 x[4,8,0] + tau[4,0] - tau[8,0] <= 999871.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,8,1]:
   1e+06 x[4,8,1] + tau[4,1] - tau[8,1] <= 999871.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,8,2]:
   1e+06 x[4,8,2] + tau[4,2] - tau[8,2] <= 999871.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,4,0]:
   1e+06 x[5,4,0] - tau[4,0] + tau[5,0] <= 999991.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,4,1]:
   1e+06 x[5,4,1] - tau[4,1] + tau[5,1] <= 999991.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,4,2]:
   1e+06 x[5,4,2] - tau[4,2] + tau[5,2] <= 999991.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,6,0]:
   1e+06 x[5,6,0] + tau[5,0] - tau[6,0] <= 999978.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,6,1]:
   1e+06 x[5,6,1] + tau[5,1] - tau[6,1] <= 999978.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,6,2]:
   1e+06 x[5,6,2] + tau[5,2] - tau[6,2] <= 999978.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,7,0]:
   1e+06 x[5,7,0] + tau[5,0] - tau[7,0] <= 999903.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,7,1]:
   1e+06 x[5,7,1] + tau[5,1] - tau[7,1] <= 999903.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,7,2]:
   1e+06 x[5,7,2] + tau[5,2] - tau[7,2] <= 999903.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,8,0]:
   1e+06 x[5,8,0] + tau[5,0] - tau[8,0] <= 999879.55
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,8,1]:
   1e+06 x[5,8,1] + tau[5,1] - tau[8,1] <= 999879.55
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,8,2]:
   1e+06 x[5,8,2] + tau[5,2] - tau[8,2] <= 999879.55
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,4,0]:
   1e+06 x[6,4,0] - tau[4,0] + tau[6,0] <= 999976.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,4,1]:
   1e+06 x[6,4,1] - tau[4,1] + tau[6,1] <= 999976.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,4,2]:
   1e+06 x[6,4,2] - tau[4,2] + tau[6,2] <= 999976.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,5,0]:
   1e+06 x[6,5,0] - tau[5,0] + tau[6,0] <= 999978.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,5,1]:
   1e+06 x[6,5,1] - tau[5,1] + tau[6,1] <= 999978.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,5,2]:
   1e+06 x[6,5,2] - tau[5,2] + tau[6,2] <= 999978.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,7,0]:
   1e+06 x[6,7,0] + tau[6,0] - tau[7,0] <= 999922.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,7,1]:
   1e+06 x[6,7,1] + tau[6,1] - tau[7,1] <= 999922.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,7,2]:
   1e+06 x[6,7,2] + tau[6,2] - tau[7,2] <= 999922.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,8,0]:
   1e+06 x[6,8,0] + tau[6,0] - tau[8,0] <= 999886.95
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,8,1]:
   1e+06 x[6,8,1] + tau[6,1] - tau[8,1] <= 999886.95
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,8,2]:
   1e+06 x[6,8,2] + tau[6,2] - tau[8,2] <= 999886.95
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,4,0]:
   1e+06 x[7,4,0] - tau[4,0] + tau[7,0] <= 999905.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,4,1]:
   1e+06 x[7,4,1] - tau[4,1] + tau[7,1] <= 999905.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,4,2]:
   1e+06 x[7,4,2] - tau[4,2] + tau[7,2] <= 999905.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,5,0]:
   1e+06 x[7,5,0] - tau[5,0] + tau[7,0] <= 999903.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,5,1]:
   1e+06 x[7,5,1] - tau[5,1] + tau[7,1] <= 999903.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,5,2]:
   1e+06 x[7,5,2] - tau[5,2] + tau[7,2] <= 999903.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,6,0]:
   1e+06 x[7,6,0] - tau[6,0] + tau[7,0] <= 999922.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,6,1]:
   1e+06 x[7,6,1] - tau[6,1] + tau[7,1] <= 999922.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,6,2]:
   1e+06 x[7,6,2] - tau[6,2] + tau[7,2] <= 999922.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,8,0]:
   1e+06 x[7,8,0] + tau[7,0] - tau[8,0] <= 999850.65
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,8,1]:
   1e+06 x[7,8,1] + tau[7,1] - tau[8,1] <= 999850.65
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,8,2]:
   1e+06 x[7,8,2] + tau[7,2] - tau[8,2] <= 999850.65
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,4,0]:
   1e+06 x[8,4,0] - tau[4,0] + tau[8,0] <= 999871.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,4,1]:
   1e+06 x[8,4,1] - tau[4,1] + tau[8,1] <= 999871.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,4,2]:
   1e+06 x[8,4,2] - tau[4,2] + tau[8,2] <= 999871.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,5,0]:
   1e+06 x[8,5,0] - tau[5,0] + tau[8,0] <= 999879.55
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,5,1]:
   1e+06 x[8,5,1] - tau[5,1] + tau[8,1] <= 999879.55
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,5,2]:
   1e+06 x[8,5,2] - tau[5,2] + tau[8,2] <= 999879.55
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,6,0]:
   1e+06 x[8,6,0] - tau[6,0] + tau[8,0] <= 999886.95
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,6,1]:
   1e+06 x[8,6,1] - tau[6,1] + tau[8,1] <= 999886.95
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,6,2]:
   1e+06 x[8,6,2] - tau[6,2] + tau[8,2] <= 999886.95
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,7,0]:
   1e+06 x[8,7,0] - tau[7,0] + tau[8,0] <= 999850.65
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,7,1]:
   1e+06 x[8,7,1] - tau[7,1] + tau[8,1] <= 999850.65
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,7,2]:
   1e+06 x[8,7,2] - tau[7,2] + tau[8,2] <= 999850.65
 lower_bound_on_customer_[4,1,0]: 1e+06 x[1,4,0] - tau[4,0] <= 999300
 lower_bound_on_customer_[4,1,1]: 1e+06 x[1,4,1] - tau[4,1] <= 999300
 lower_bound_on_customer_[4,1,2]: 1e+06 x[1,4,2] - tau[4,2] <= 999300
 lower_bound_on_customer_[4,2,0]: 1e+06 x[2,4,0] - tau[4,0] <= 999300
 lower_bound_on_customer_[4,2,1]: 1e+06 x[2,4,1] - tau[4,1] <= 999300
 lower_bound_on_customer_[4,2,2]: 1e+06 x[2,4,2] - tau[4,2] <= 999300
 lower_bound_on_customer_[4,3,0]: 1e+06 x[3,4,0] - tau[4,0] <= 999300
 lower_bound_on_customer_[4,3,1]: 1e+06 x[3,4,1] - tau[4,1] <= 999300
 lower_bound_on_customer_[4,3,2]: 1e+06 x[3,4,2] - tau[4,2] <= 999300
 lower_bound_on_customer_[4,5,0]: 1e+06 x[5,4,0] - tau[4,0] <= 999300
 lower_bound_on_customer_[4,5,1]: 1e+06 x[5,4,1] - tau[4,1] <= 999300
 lower_bound_on_customer_[4,5,2]: 1e+06 x[5,4,2] - tau[4,2] <= 999300
 lower_bound_on_customer_[4,6,0]: 1e+06 x[6,4,0] - tau[4,0] <= 999300
 lower_bound_on_customer_[4,6,1]: 1e+06 x[6,4,1] - tau[4,1] <= 999300
 lower_bound_on_customer_[4,6,2]: 1e+06 x[6,4,2] - tau[4,2] <= 999300
 lower_bound_on_customer_[4,7,0]: 1e+06 x[7,4,0] - tau[4,0] <= 999300
 lower_bound_on_customer_[4,7,1]: 1e+06 x[7,4,1] - tau[4,1] <= 999300
 lower_bound_on_customer_[4,7,2]: 1e+06 x[7,4,2] - tau[4,2] <= 999300
 lower_bound_on_customer_[4,8,0]: 1e+06 x[8,4,0] - tau[4,0] <= 999300
 lower_bound_on_customer_[4,8,1]: 1e+06 x[8,4,1] - tau[4,1] <= 999300
 lower_bound_on_customer_[4,8,2]: 1e+06 x[8,4,2] - tau[4,2] <= 999300
 lower_bound_on_customer_[5,1,0]: 1e+06 x[1,5,0] - tau[5,0] <= 999300
 lower_bound_on_customer_[5,1,1]: 1e+06 x[1,5,1] - tau[5,1] <= 999300
 lower_bound_on_customer_[5,1,2]: 1e+06 x[1,5,2] - tau[5,2] <= 999300
 lower_bound_on_customer_[5,2,0]: 1e+06 x[2,5,0] - tau[5,0] <= 999300
 lower_bound_on_customer_[5,2,1]: 1e+06 x[2,5,1] - tau[5,1] <= 999300
 lower_bound_on_customer_[5,2,2]: 1e+06 x[2,5,2] - tau[5,2] <= 999300
 lower_bound_on_customer_[5,3,0]: 1e+06 x[3,5,0] - tau[5,0] <= 999300
 lower_bound_on_customer_[5,3,1]: 1e+06 x[3,5,1] - tau[5,1] <= 999300
 lower_bound_on_customer_[5,3,2]: 1e+06 x[3,5,2] - tau[5,2] <= 999300
 lower_bound_on_customer_[5,4,0]: 1e+06 x[4,5,0] - tau[5,0] <= 999300
 lower_bound_on_customer_[5,4,1]: 1e+06 x[4,5,1] - tau[5,1] <= 999300
 lower_bound_on_customer_[5,4,2]: 1e+06 x[4,5,2] - tau[5,2] <= 999300
 lower_bound_on_customer_[5,6,0]: 1e+06 x[6,5,0] - tau[5,0] <= 999300
 lower_bound_on_customer_[5,6,1]: 1e+06 x[6,5,1] - tau[5,1] <= 999300
 lower_bound_on_customer_[5,6,2]: 1e+06 x[6,5,2] - tau[5,2] <= 999300
 lower_bound_on_customer_[5,7,0]: 1e+06 x[7,5,0] - tau[5,0] <= 999300
 lower_bound_on_customer_[5,7,1]: 1e+06 x[7,5,1] - tau[5,1] <= 999300
 lower_bound_on_customer_[5,7,2]: 1e+06 x[7,5,2] - tau[5,2] <= 999300
 lower_bound_on_customer_[5,8,0]: 1e+06 x[8,5,0] - tau[5,0] <= 999300
 lower_bound_on_customer_[5,8,1]: 1e+06 x[8,5,1] - tau[5,1] <= 999300
 lower_bound_on_customer_[5,8,2]: 1e+06 x[8,5,2] - tau[5,2] <= 999300
 lower_bound_on_customer_[6,1,0]: 1e+06 x[1,6,0] - tau[6,0] <= 999300
 lower_bound_on_customer_[6,1,1]: 1e+06 x[1,6,1] - tau[6,1] <= 999300
 lower_bound_on_customer_[6,1,2]: 1e+06 x[1,6,2] - tau[6,2] <= 999300
 lower_bound_on_customer_[6,2,0]: 1e+06 x[2,6,0] - tau[6,0] <= 999300
 lower_bound_on_customer_[6,2,1]: 1e+06 x[2,6,1] - tau[6,1] <= 999300
 lower_bound_on_customer_[6,2,2]: 1e+06 x[2,6,2] - tau[6,2] <= 999300
 lower_bound_on_customer_[6,3,0]: 1e+06 x[3,6,0] - tau[6,0] <= 999300
 lower_bound_on_customer_[6,3,1]: 1e+06 x[3,6,1] - tau[6,1] <= 999300
 lower_bound_on_customer_[6,3,2]: 1e+06 x[3,6,2] - tau[6,2] <= 999300
 lower_bound_on_customer_[6,4,0]: 1e+06 x[4,6,0] - tau[6,0] <= 999300
 lower_bound_on_customer_[6,4,1]: 1e+06 x[4,6,1] - tau[6,1] <= 999300
 lower_bound_on_customer_[6,4,2]: 1e+06 x[4,6,2] - tau[6,2] <= 999300
 lower_bound_on_customer_[6,5,0]: 1e+06 x[5,6,0] - tau[6,0] <= 999300
 lower_bound_on_customer_[6,5,1]: 1e+06 x[5,6,1] - tau[6,1] <= 999300
 lower_bound_on_customer_[6,5,2]: 1e+06 x[5,6,2] - tau[6,2] <= 999300
 lower_bound_on_customer_[6,7,0]: 1e+06 x[7,6,0] - tau[6,0] <= 999300
 lower_bound_on_customer_[6,7,1]: 1e+06 x[7,6,1] - tau[6,1] <= 999300
 lower_bound_on_customer_[6,7,2]: 1e+06 x[7,6,2] - tau[6,2] <= 999300
 lower_bound_on_customer_[6,8,0]: 1e+06 x[8,6,0] - tau[6,0] <= 999300
 lower_bound_on_customer_[6,8,1]: 1e+06 x[8,6,1] - tau[6,1] <= 999300
 lower_bound_on_customer_[6,8,2]: 1e+06 x[8,6,2] - tau[6,2] <= 999300
 lower_bound_on_customer_[7,1,0]: 1e+06 x[1,7,0] - tau[7,0] <= 999300
 lower_bound_on_customer_[7,1,1]: 1e+06 x[1,7,1] - tau[7,1] <= 999300
 lower_bound_on_customer_[7,1,2]: 1e+06 x[1,7,2] - tau[7,2] <= 999300
 lower_bound_on_customer_[7,2,0]: 1e+06 x[2,7,0] - tau[7,0] <= 999300
 lower_bound_on_customer_[7,2,1]: 1e+06 x[2,7,1] - tau[7,1] <= 999300
 lower_bound_on_customer_[7,2,2]: 1e+06 x[2,7,2] - tau[7,2] <= 999300
 lower_bound_on_customer_[7,3,0]: 1e+06 x[3,7,0] - tau[7,0] <= 999300
 lower_bound_on_customer_[7,3,1]: 1e+06 x[3,7,1] - tau[7,1] <= 999300
 lower_bound_on_customer_[7,3,2]: 1e+06 x[3,7,2] - tau[7,2] <= 999300
 lower_bound_on_customer_[7,4,0]: 1e+06 x[4,7,0] - tau[7,0] <= 999300
 lower_bound_on_customer_[7,4,1]: 1e+06 x[4,7,1] - tau[7,1] <= 999300
 lower_bound_on_customer_[7,4,2]: 1e+06 x[4,7,2] - tau[7,2] <= 999300
 lower_bound_on_customer_[7,5,0]: 1e+06 x[5,7,0] - tau[7,0] <= 999300
 lower_bound_on_customer_[7,5,1]: 1e+06 x[5,7,1] - tau[7,1] <= 999300
 lower_bound_on_customer_[7,5,2]: 1e+06 x[5,7,2] - tau[7,2] <= 999300
 lower_bound_on_customer_[7,6,0]: 1e+06 x[6,7,0] - tau[7,0] <= 999300
 lower_bound_on_customer_[7,6,1]: 1e+06 x[6,7,1] - tau[7,1] <= 999300
 lower_bound_on_customer_[7,6,2]: 1e+06 x[6,7,2] - tau[7,2] <= 999300
 lower_bound_on_customer_[7,8,0]: 1e+06 x[8,7,0] - tau[7,0] <= 999300
 lower_bound_on_customer_[7,8,1]: 1e+06 x[8,7,1] - tau[7,1] <= 999300
 lower_bound_on_customer_[7,8,2]: 1e+06 x[8,7,2] - tau[7,2] <= 999300
 lower_bound_on_customer_[8,1,0]: 1e+06 x[1,8,0] - tau[8,0] <= 999300
 lower_bound_on_customer_[8,1,1]: 1e+06 x[1,8,1] - tau[8,1] <= 999300
 lower_bound_on_customer_[8,1,2]: 1e+06 x[1,8,2] - tau[8,2] <= 999300
 lower_bound_on_customer_[8,2,0]: 1e+06 x[2,8,0] - tau[8,0] <= 999300
 lower_bound_on_customer_[8,2,1]: 1e+06 x[2,8,1] - tau[8,1] <= 999300
 lower_bound_on_customer_[8,2,2]: 1e+06 x[2,8,2] - tau[8,2] <= 999300
 lower_bound_on_customer_[8,3,0]: 1e+06 x[3,8,0] - tau[8,0] <= 999300
 lower_bound_on_customer_[8,3,1]: 1e+06 x[3,8,1] - tau[8,1] <= 999300
 lower_bound_on_customer_[8,3,2]: 1e+06 x[3,8,2] - tau[8,2] <= 999300
 lower_bound_on_customer_[8,4,0]: 1e+06 x[4,8,0] - tau[8,0] <= 999300
 lower_bound_on_customer_[8,4,1]: 1e+06 x[4,8,1] - tau[8,1] <= 999300
 lower_bound_on_customer_[8,4,2]: 1e+06 x[4,8,2] - tau[8,2] <= 999300
 lower_bound_on_customer_[8,5,0]: 1e+06 x[5,8,0] - tau[8,0] <= 999300
 lower_bound_on_customer_[8,5,1]: 1e+06 x[5,8,1] - tau[8,1] <= 999300
 lower_bound_on_customer_[8,5,2]: 1e+06 x[5,8,2] - tau[8,2] <= 999300
 lower_bound_on_customer_[8,6,0]: 1e+06 x[6,8,0] - tau[8,0] <= 999300
 lower_bound_on_customer_[8,6,1]: 1e+06 x[6,8,1] - tau[8,1] <= 999300
 lower_bound_on_customer_[8,6,2]: 1e+06 x[6,8,2] - tau[8,2] <= 999300
 lower_bound_on_customer_[8,7,0]: 1e+06 x[7,8,0] - tau[8,0] <= 999300
 lower_bound_on_customer_[8,7,1]: 1e+06 x[7,8,1] - tau[8,1] <= 999300
 lower_bound_on_customer_[8,7,2]: 1e+06 x[7,8,2] - tau[8,2] <= 999300
 upper_bound_on_customer_[4,1,0]: - 1e+06 x[1,4,0] - tau[4,0]
   >= -1.0013e+06
 upper_bound_on_customer_[4,1,1]: - 1e+06 x[1,4,1] - tau[4,1]
   >= -1.0013e+06
 upper_bound_on_customer_[4,1,2]: - 1e+06 x[1,4,2] - tau[4,2]
   >= -1.0013e+06
 upper_bound_on_customer_[4,2,0]: - 1e+06 x[2,4,0] - tau[4,0]
   >= -1.0013e+06
 upper_bound_on_customer_[4,2,1]: - 1e+06 x[2,4,1] - tau[4,1]
   >= -1.0013e+06
 upper_bound_on_customer_[4,2,2]: - 1e+06 x[2,4,2] - tau[4,2]
   >= -1.0013e+06
 upper_bound_on_customer_[4,3,0]: - 1e+06 x[3,4,0] - tau[4,0]
   >= -1.0013e+06
 upper_bound_on_customer_[4,3,1]: - 1e+06 x[3,4,1] - tau[4,1]
   >= -1.0013e+06
 upper_bound_on_customer_[4,3,2]: - 1e+06 x[3,4,2] - tau[4,2]
   >= -1.0013e+06
 upper_bound_on_customer_[4,5,0]: - 1e+06 x[5,4,0] - tau[4,0]
   >= -1.0013e+06
 upper_bound_on_customer_[4,5,1]: - 1e+06 x[5,4,1] - tau[4,1]
   >= -1.0013e+06
 upper_bound_on_customer_[4,5,2]: - 1e+06 x[5,4,2] - tau[4,2]
   >= -1.0013e+06
 upper_bound_on_customer_[4,6,0]: - 1e+06 x[6,4,0] - tau[4,0]
   >= -1.0013e+06
 upper_bound_on_customer_[4,6,1]: - 1e+06 x[6,4,1] - tau[4,1]
   >= -1.0013e+06
 upper_bound_on_customer_[4,6,2]: - 1e+06 x[6,4,2] - tau[4,2]
   >= -1.0013e+06
 upper_bound_on_customer_[4,7,0]: - 1e+06 x[7,4,0] - tau[4,0]
   >= -1.0013e+06
 upper_bound_on_customer_[4,7,1]: - 1e+06 x[7,4,1] - tau[4,1]
   >= -1.0013e+06
 upper_bound_on_customer_[4,7,2]: - 1e+06 x[7,4,2] - tau[4,2]
   >= -1.0013e+06
 upper_bound_on_customer_[4,8,0]: - 1e+06 x[8,4,0] - tau[4,0]
   >= -1.0013e+06
 upper_bound_on_customer_[4,8,1]: - 1e+06 x[8,4,1] - tau[4,1]
   >= -1.0013e+06
 upper_bound_on_customer_[4,8,2]: - 1e+06 x[8,4,2] - tau[4,2]
   >= -1.0013e+06
 upper_bound_on_customer_[5,1,0]: - 1e+06 x[1,5,0] - tau[5,0]
   >= -1.0014e+06
 upper_bound_on_customer_[5,1,1]: - 1e+06 x[1,5,1] - tau[5,1]
   >= -1.0014e+06
 upper_bound_on_customer_[5,1,2]: - 1e+06 x[1,5,2] - tau[5,2]
   >= -1.0014e+06
 upper_bound_on_customer_[5,2,0]: - 1e+06 x[2,5,0] - tau[5,0]
   >= -1.0014e+06
 upper_bound_on_customer_[5,2,1]: - 1e+06 x[2,5,1] - tau[5,1]
   >= -1.0014e+06
 upper_bound_on_customer_[5,2,2]: - 1e+06 x[2,5,2] - tau[5,2]
   >= -1.0014e+06
 upper_bound_on_customer_[5,3,0]: - 1e+06 x[3,5,0] - tau[5,0]
   >= -1.0014e+06
 upper_bound_on_customer_[5,3,1]: - 1e+06 x[3,5,1] - tau[5,1]
   >= -1.0014e+06
 upper_bound_on_customer_[5,3,2]: - 1e+06 x[3,5,2] - tau[5,2]
   >= -1.0014e+06
 upper_bound_on_customer_[5,4,0]: - 1e+06 x[4,5,0] - tau[5,0]
   >= -1.0014e+06
 upper_bound_on_customer_[5,4,1]: - 1e+06 x[4,5,1] - tau[5,1]
   >= -1.0014e+06
 upper_bound_on_customer_[5,4,2]: - 1e+06 x[4,5,2] - tau[5,2]
   >= -1.0014e+06
 upper_bound_on_customer_[5,6,0]: - 1e+06 x[6,5,0] - tau[5,0]
   >= -1.0014e+06
 upper_bound_on_customer_[5,6,1]: - 1e+06 x[6,5,1] - tau[5,1]
   >= -1.0014e+06
 upper_bound_on_customer_[5,6,2]: - 1e+06 x[6,5,2] - tau[5,2]
   >= -1.0014e+06
 upper_bound_on_customer_[5,7,0]: - 1e+06 x[7,5,0] - tau[5,0]
   >= -1.0014e+06
 upper_bound_on_customer_[5,7,1]: - 1e+06 x[7,5,1] - tau[5,1]
   >= -1.0014e+06
 upper_bound_on_customer_[5,7,2]: - 1e+06 x[7,5,2] - tau[5,2]
   >= -1.0014e+06
 upper_bound_on_customer_[5,8,0]: - 1e+06 x[8,5,0] - tau[5,0]
   >= -1.0014e+06
 upper_bound_on_customer_[5,8,1]: - 1e+06 x[8,5,1] - tau[5,1]
   >= -1.0014e+06
 upper_bound_on_customer_[5,8,2]: - 1e+06 x[8,5,2] - tau[5,2]
   >= -1.0014e+06
 upper_bound_on_customer_[6,1,0]: - 1e+06 x[1,6,0] - tau[6,0]
   >= -1.0015e+06
 upper_bound_on_customer_[6,1,1]: - 1e+06 x[1,6,1] - tau[6,1]
   >= -1.0015e+06
 upper_bound_on_customer_[6,1,2]: - 1e+06 x[1,6,2] - tau[6,2]
   >= -1.0015e+06
 upper_bound_on_customer_[6,2,0]: - 1e+06 x[2,6,0] - tau[6,0]
   >= -1.0015e+06
 upper_bound_on_customer_[6,2,1]: - 1e+06 x[2,6,1] - tau[6,1]
   >= -1.0015e+06
 upper_bound_on_customer_[6,2,2]: - 1e+06 x[2,6,2] - tau[6,2]
   >= -1.0015e+06
 upper_bound_on_customer_[6,3,0]: - 1e+06 x[3,6,0] - tau[6,0]
   >= -1.0015e+06
 upper_bound_on_customer_[6,3,1]: - 1e+06 x[3,6,1] - tau[6,1]
   >= -1.0015e+06
 upper_bound_on_customer_[6,3,2]: - 1e+06 x[3,6,2] - tau[6,2]
   >= -1.0015e+06
 upper_bound_on_customer_[6,4,0]: - 1e+06 x[4,6,0] - tau[6,0]
   >= -1.0015e+06
 upper_bound_on_customer_[6,4,1]: - 1e+06 x[4,6,1] - tau[6,1]
   >= -1.0015e+06
 upper_bound_on_customer_[6,4,2]: - 1e+06 x[4,6,2] - tau[6,2]
   >= -1.0015e+06
 upper_bound_on_customer_[6,5,0]: - 1e+06 x[5,6,0] - tau[6,0]
   >= -1.0015e+06
 upper_bound_on_customer_[6,5,1]: - 1e+06 x[5,6,1] - tau[6,1]
   >= -1.0015e+06
 upper_bound_on_customer_[6,5,2]: - 1e+06 x[5,6,2] - tau[6,2]
   >= -1.0015e+06
 upper_bound_on_customer_[6,7,0]: - 1e+06 x[7,6,0] - tau[6,0]
   >= -1.0015e+06
 upper_bound_on_customer_[6,7,1]: - 1e+06 x[7,6,1] - tau[6,1]
   >= -1.0015e+06
 upper_bound_on_customer_[6,7,2]: - 1e+06 x[7,6,2] - tau[6,2]
   >= -1.0015e+06
 upper_bound_on_customer_[6,8,0]: - 1e+06 x[8,6,0] - tau[6,0]
   >= -1.0015e+06
 upper_bound_on_customer_[6,8,1]: - 1e+06 x[8,6,1] - tau[6,1]
   >= -1.0015e+06
 upper_bound_on_customer_[6,8,2]: - 1e+06 x[8,6,2] - tau[6,2]
   >= -1.0015e+06
 upper_bound_on_customer_[7,1,0]: - 1e+06 x[1,7,0] - tau[7,0]
   >= -1.0016e+06
 upper_bound_on_customer_[7,1,1]: - 1e+06 x[1,7,1] - tau[7,1]
   >= -1.0016e+06
 upper_bound_on_customer_[7,1,2]: - 1e+06 x[1,7,2] - tau[7,2]
   >= -1.0016e+06
 upper_bound_on_customer_[7,2,0]: - 1e+06 x[2,7,0] - tau[7,0]
   >= -1.0016e+06
 upper_bound_on_customer_[7,2,1]: - 1e+06 x[2,7,1] - tau[7,1]
   >= -1.0016e+06
 upper_bound_on_customer_[7,2,2]: - 1e+06 x[2,7,2] - tau[7,2]
   >= -1.0016e+06
 upper_bound_on_customer_[7,3,0]: - 1e+06 x[3,7,0] - tau[7,0]
   >= -1.0016e+06
 upper_bound_on_customer_[7,3,1]: - 1e+06 x[3,7,1] - tau[7,1]
   >= -1.0016e+06
 upper_bound_on_customer_[7,3,2]: - 1e+06 x[3,7,2] - tau[7,2]
   >= -1.0016e+06
 upper_bound_on_customer_[7,4,0]: - 1e+06 x[4,7,0] - tau[7,0]
   >= -1.0016e+06
 upper_bound_on_customer_[7,4,1]: - 1e+06 x[4,7,1] - tau[7,1]
   >= -1.0016e+06
 upper_bound_on_customer_[7,4,2]: - 1e+06 x[4,7,2] - tau[7,2]
   >= -1.0016e+06
 upper_bound_on_customer_[7,5,0]: - 1e+06 x[5,7,0] - tau[7,0]
   >= -1.0016e+06
 upper_bound_on_customer_[7,5,1]: - 1e+06 x[5,7,1] - tau[7,1]
   >= -1.0016e+06
 upper_bound_on_customer_[7,5,2]: - 1e+06 x[5,7,2] - tau[7,2]
   >= -1.0016e+06
 upper_bound_on_customer_[7,6,0]: - 1e+06 x[6,7,0] - tau[7,0]
   >= -1.0016e+06
 upper_bound_on_customer_[7,6,1]: - 1e+06 x[6,7,1] - tau[7,1]
   >= -1.0016e+06
 upper_bound_on_customer_[7,6,2]: - 1e+06 x[6,7,2] - tau[7,2]
   >= -1.0016e+06
 upper_bound_on_customer_[7,8,0]: - 1e+06 x[8,7,0] - tau[7,0]
   >= -1.0016e+06
 upper_bound_on_customer_[7,8,1]: - 1e+06 x[8,7,1] - tau[7,1]
   >= -1.0016e+06
 upper_bound_on_customer_[7,8,2]: - 1e+06 x[8,7,2] - tau[7,2]
   >= -1.0016e+06
 upper_bound_on_customer_[8,1,0]: - 1e+06 x[1,8,0] - tau[8,0]
   >= -1.0017e+06
 upper_bound_on_customer_[8,1,1]: - 1e+06 x[1,8,1] - tau[8,1]
   >= -1.0017e+06
 upper_bound_on_customer_[8,1,2]: - 1e+06 x[1,8,2] - tau[8,2]
   >= -1.0017e+06
 upper_bound_on_customer_[8,2,0]: - 1e+06 x[2,8,0] - tau[8,0]
   >= -1.0017e+06
 upper_bound_on_customer_[8,2,1]: - 1e+06 x[2,8,1] - tau[8,1]
   >= -1.0017e+06
 upper_bound_on_customer_[8,2,2]: - 1e+06 x[2,8,2] - tau[8,2]
   >= -1.0017e+06
 upper_bound_on_customer_[8,3,0]: - 1e+06 x[3,8,0] - tau[8,0]
   >= -1.0017e+06
 upper_bound_on_customer_[8,3,1]: - 1e+06 x[3,8,1] - tau[8,1]
   >= -1.0017e+06
 upper_bound_on_customer_[8,3,2]: - 1e+06 x[3,8,2] - tau[8,2]
   >= -1.0017e+06
 upper_bound_on_customer_[8,4,0]: - 1e+06 x[4,8,0] - tau[8,0]
   >= -1.0017e+06
 upper_bound_on_customer_[8,4,1]: - 1e+06 x[4,8,1] - tau[8,1]
   >= -1.0017e+06
 upper_bound_on_customer_[8,4,2]: - 1e+06 x[4,8,2] - tau[8,2]
   >= -1.0017e+06
 upper_bound_on_customer_[8,5,0]: - 1e+06 x[5,8,0] - tau[8,0]
   >= -1.0017e+06
 upper_bound_on_customer_[8,5,1]: - 1e+06 x[5,8,1] - tau[8,1]
   >= -1.0017e+06
 upper_bound_on_customer_[8,5,2]: - 1e+06 x[5,8,2] - tau[8,2]
   >= -1.0017e+06
 upper_bound_on_customer_[8,6,0]: - 1e+06 x[6,8,0] - tau[8,0]
   >= -1.0017e+06
 upper_bound_on_customer_[8,6,1]: - 1e+06 x[6,8,1] - tau[8,1]
   >= -1.0017e+06
 upper_bound_on_customer_[8,6,2]: - 1e+06 x[6,8,2] - tau[8,2]
   >= -1.0017e+06
 upper_bound_on_customer_[8,7,0]: - 1e+06 x[7,8,0] - tau[8,0]
   >= -1.0017e+06
 upper_bound_on_customer_[8,7,1]: - 1e+06 x[7,8,1] - tau[8,1]
   >= -1.0017e+06
 upper_bound_on_customer_[8,7,2]: - 1e+06 x[7,8,2] - tau[8,2]
   >= -1.0017e+06
 max_endurance_of_drone[0]: - launch[0] + land[0] <= 10800
 max_endurance_of_drone[1]: - launch[1] + land[1] <= 10800
 max_endurance_of_drone[2]: - launch[2] + land[2] <= 10800
 Capacity[0,1]: 2 x[1,4,0] + 2 x[1,5,0] + 2 x[1,6,0] + 2 x[1,7,0]
   + 2 x[1,8,0] + 2 x[4,5,0] + 2 x[4,6,0] + 2 x[4,7,0] + 2 x[4,8,0]
   + 2 x[5,4,0] + 2 x[5,6,0] + 2 x[5,7,0] + 2 x[5,8,0] + 2 x[6,4,0]
   + 2 x[6,5,0] + 2 x[6,7,0] + 2 x[6,8,0] + 2 x[7,4,0] + 2 x[7,5,0]
   + 2 x[7,6,0] + 2 x[7,8,0] + 2 x[8,4,0] + 2 x[8,5,0] + 2 x[8,6,0]
   + 2 x[8,7,0] <= 10
 Capacity[0,2]: 2 x[2,4,0] + 2 x[2,5,0] + 2 x[2,6,0] + 2 x[2,7,0]
   + 2 x[2,8,0] + 2 x[4,5,0] + 2 x[4,6,0] + 2 x[4,7,0] + 2 x[4,8,0]
   + 2 x[5,4,0] + 2 x[5,6,0] + 2 x[5,7,0] + 2 x[5,8,0] + 2 x[6,4,0]
   + 2 x[6,5,0] + 2 x[6,7,0] + 2 x[6,8,0] + 2 x[7,4,0] + 2 x[7,5,0]
   + 2 x[7,6,0] + 2 x[7,8,0] + 2 x[8,4,0] + 2 x[8,5,0] + 2 x[8,6,0]
   + 2 x[8,7,0] <= 10
 Capacity[0,3]: 2 x[3,4,0] + 2 x[3,5,0] + 2 x[3,6,0] + 2 x[3,7,0]
   + 2 x[3,8,0] + 2 x[4,5,0] + 2 x[4,6,0] + 2 x[4,7,0] + 2 x[4,8,0]
   + 2 x[5,4,0] + 2 x[5,6,0] + 2 x[5,7,0] + 2 x[5,8,0] + 2 x[6,4,0]
   + 2 x[6,5,0] + 2 x[6,7,0] + 2 x[6,8,0] + 2 x[7,4,0] + 2 x[7,5,0]
   + 2 x[7,6,0] + 2 x[7,8,0] + 2 x[8,4,0] + 2 x[8,5,0] + 2 x[8,6,0]
   + 2 x[8,7,0] <= 10
 Capacity[1,1]: 2 x[1,4,1] + 2 x[1,5,1] + 2 x[1,6,1] + 2 x[1,7,1]
   + 2 x[1,8,1] + 2 x[4,5,1] + 2 x[4,6,1] + 2 x[4,7,1] + 2 x[4,8,1]
   + 2 x[5,4,1] + 2 x[5,6,1] + 2 x[5,7,1] + 2 x[5,8,1] + 2 x[6,4,1]
   + 2 x[6,5,1] + 2 x[6,7,1] + 2 x[6,8,1] + 2 x[7,4,1] + 2 x[7,5,1]
   + 2 x[7,6,1] + 2 x[7,8,1] + 2 x[8,4,1] + 2 x[8,5,1] + 2 x[8,6,1]
   + 2 x[8,7,1] <= 10
 Capacity[1,2]: 2 x[2,4,1] + 2 x[2,5,1] + 2 x[2,6,1] + 2 x[2,7,1]
   + 2 x[2,8,1] + 2 x[4,5,1] + 2 x[4,6,1] + 2 x[4,7,1] + 2 x[4,8,1]
   + 2 x[5,4,1] + 2 x[5,6,1] + 2 x[5,7,1] + 2 x[5,8,1] + 2 x[6,4,1]
   + 2 x[6,5,1] + 2 x[6,7,1] + 2 x[6,8,1] + 2 x[7,4,1] + 2 x[7,5,1]
   + 2 x[7,6,1] + 2 x[7,8,1] + 2 x[8,4,1] + 2 x[8,5,1] + 2 x[8,6,1]
   + 2 x[8,7,1] <= 10
 Capacity[1,3]: 2 x[3,4,1] + 2 x[3,5,1] + 2 x[3,6,1] + 2 x[3,7,1]
   + 2 x[3,8,1] + 2 x[4,5,1] + 2 x[4,6,1] + 2 x[4,7,1] + 2 x[4,8,1]
   + 2 x[5,4,1] + 2 x[5,6,1] + 2 x[5,7,1] + 2 x[5,8,1] + 2 x[6,4,1]
   + 2 x[6,5,1] + 2 x[6,7,1] + 2 x[6,8,1] + 2 x[7,4,1] + 2 x[7,5,1]
   + 2 x[7,6,1] + 2 x[7,8,1] + 2 x[8,4,1] + 2 x[8,5,1] + 2 x[8,6,1]
   + 2 x[8,7,1] <= 10
 Capacity[2,1]: 2 x[1,4,2] + 2 x[1,5,2] + 2 x[1,6,2] + 2 x[1,7,2]
   + 2 x[1,8,2] + 2 x[4,5,2] + 2 x[4,6,2] + 2 x[4,7,2] + 2 x[4,8,2]
   + 2 x[5,4,2] + 2 x[5,6,2] + 2 x[5,7,2] + 2 x[5,8,2] + 2 x[6,4,2]
   + 2 x[6,5,2] + 2 x[6,7,2] + 2 x[6,8,2] + 2 x[7,4,2] + 2 x[7,5,2]
   + 2 x[7,6,2] + 2 x[7,8,2] + 2 x[8,4,2] + 2 x[8,5,2] + 2 x[8,6,2]
   + 2 x[8,7,2] <= 10
 Capacity[2,2]: 2 x[2,4,2] + 2 x[2,5,2] + 2 x[2,6,2] + 2 x[2,7,2]
   + 2 x[2,8,2] + 2 x[4,5,2] + 2 x[4,6,2] + 2 x[4,7,2] + 2 x[4,8,2]
   + 2 x[5,4,2] + 2 x[5,6,2] + 2 x[5,7,2] + 2 x[5,8,2] + 2 x[6,4,2]
   + 2 x[6,5,2] + 2 x[6,7,2] + 2 x[6,8,2] + 2 x[7,4,2] + 2 x[7,5,2]
   + 2 x[7,6,2] + 2 x[7,8,2] + 2 x[8,4,2] + 2 x[8,5,2] + 2 x[8,6,2]
   + 2 x[8,7,2] <= 10
 Capacity[2,3]: 2 x[3,4,2] + 2 x[3,5,2] + 2 x[3,6,2] + 2 x[3,7,2]
   + 2 x[3,8,2] + 2 x[4,5,2] + 2 x[4,6,2] + 2 x[4,7,2] + 2 x[4,8,2]
   + 2 x[5,4,2] + 2 x[5,6,2] + 2 x[5,7,2] + 2 x[5,8,2] + 2 x[6,4,2]
   + 2 x[6,5,2] + 2 x[6,7,2] + 2 x[6,8,2] + 2 x[7,4,2] + 2 x[7,5,2]
   + 2 x[7,6,2] + 2 x[7,8,2] + 2 x[8,4,2] + 2 x[8,5,2] + 2 x[8,6,2]
   + 2 x[8,7,2] <= 10
 launch_time[0]: - 82 x[0,1,0] - 110.4 x[0,2,0] - 123.6 x[0,3,0]
   - launch[0] + [ x[0,1,0] * tau[1,0] + x[0,2,0] * tau[2,0]
   + x[0,3,0] * tau[3,0] ] >= 0
 launch_time[1]: - 82 x[0,1,1] - 110.4 x[0,2,1] - 123.6 x[0,3,1]
   - launch[1] + [ x[0,1,1] * tau[1,1] + x[0,2,1] * tau[2,1]
   + x[0,3,1] * tau[3,1] ] >= 0
 launch_time[2]: - 82 x[0,1,2] - 110.4 x[0,2,2] - 123.6 x[0,3,2]
   - launch[2] + [ x[0,1,2] * tau[1,2] + x[0,2,2] * tau[2,2]
   + x[0,3,2] * tau[3,2] ] >= 0
 land_time[0]: 241.1 x[4,0,0] + 236.85 x[5,0,0] + 217.55 x[6,0,0]
   + 181.7 x[7,0,0] + 166.3 x[8,0,0] - land[0] + [ x[4,0,0] * tau[4,0]
   + x[5,0,0] * tau[5,0] + x[6,0,0] * tau[6,0] + x[7,0,0] * tau[7,0]
   + x[8,0,0] * tau[8,0] ] <= 0
 land_time[1]: 241.1 x[4,0,1] + 236.85 x[5,0,1] + 217.55 x[6,0,1]
   + 181.7 x[7,0,1] + 166.3 x[8,0,1] - land[1] + [ x[4,0,1] * tau[4,1]
   + x[5,0,1] * tau[5,1] + x[6,0,1] * tau[6,1] + x[7,0,1] * tau[7,1]
   + x[8,0,1] * tau[8,1] ] <= 0
 land_time[2]: 241.1 x[4,0,2] + 236.85 x[5,0,2] + 217.55 x[6,0,2]
   + 181.7 x[7,0,2] + 166.3 x[8,0,2] - land[2] + [ x[4,0,2] * tau[4,2]
   + x[5,0,2] * tau[5,2] + x[6,0,2] * tau[6,2] + x[7,0,2] * tau[7,2]
   + x[8,0,2] * tau[8,2] ] <= 0
 either_launch_time_of_drone_m_is_D_time_after_launch_time_of_drone_k[0,1]:
   launch[0] - launch[1] - 1e+06 y_1[0,1] + [ 10 x_k[0] * x_k[1] ] <= 0
 either_launch_time_of_drone_m_is_D_time_after_launch_time_of_drone_k[0,2]:
   launch[0] - launch[2] - 1e+06 y_1[0,2] + [ 10 x_k[0] * x_k[2] ] <= 0
 either_launch_time_of_drone_m_is_D_time_after_launch_time_of_drone_k[1,2]:
   launch[1] - launch[2] - 1e+06 y_1[1,2] + [ 10 x_k[1] * x_k[2] ] <= 0
 or_launch_time_of_drone_k_is_D_time_after_launch_time_of_drone_m[0,1]:
   launch[0] - launch[1] + 1e+06 y_1[0,1] + [ 10 x_k[0] * x_k[1] ]
   <= 1e+06
 or_launch_time_of_drone_k_is_D_time_after_launch_time_of_drone_m[0,2]:
   launch[0] - launch[2] + 1e+06 y_1[0,2] + [ 10 x_k[0] * x_k[2] ]
   <= 1e+06
 or_launch_time_of_drone_k_is_D_time_after_launch_time_of_drone_m[1,2]:
   launch[1] - launch[2] + 1e+06 y_1[1,2] + [ 10 x_k[1] * x_k[2] ]
   <= 1e+06
 either_land_time_of_drone_m_is_D_time_after_land_time_of_drone_k[0,1]:
   land[0] - land[1] - 1e+06 y_2[0,1] + [ 10 x_k[0] * x_k[1] ] <= 0
 either_land_time_of_drone_m_is_D_time_after_land_time_of_drone_k[0,2]:
   land[0] - land[2] - 1e+06 y_2[0,2] + [ 10 x_k[0] * x_k[2] ] <= 0
 either_land_time_of_drone_m_is_D_time_after_land_time_of_drone_k[1,2]:
   land[1] - land[2] - 1e+06 y_2[1,2] + [ 10 x_k[1] * x_k[2] ] <= 0
 or_land_time_of_drone_k_is_D_time_after_land_time_of_drone_m[0,1]: land[0]
   - land[1] + 1e+06 y_2[0,1] + [ 10 x_k[0] * x_k[1] ] <= 1e+06
 or_land_time_of_drone_k_is_D_time_after_land_time_of_drone_m[0,2]: land[0]
   - land[2] + 1e+06 y_2[0,2] + [ 10 x_k[0] * x_k[2] ] <= 1e+06
 or_land_time_of_drone_k_is_D_time_after_land_time_of_drone_m[1,2]: land[1]
   - land[2] + 1e+06 y_2[1,2] + [ 10 x_k[1] * x_k[2] ] <= 1e+06
Bounds
 Constant = 1
Binaries
 x[0,1,0] x[0,1,1] x[0,1,2] x[0,2,0] x[0,2,1] x[0,2,2] x[0,3,0] x[0,3,1]
 x[0,3,2] x[1,4,0] x[1,4,1] x[1,4,2] x[1,5,0] x[1,5,1] x[1,5,2] x[1,6,0]
 x[1,6,1] x[1,6,2] x[1,7,0] x[1,7,1] x[1,7,2] x[1,8,0] x[1,8,1] x[1,8,2]
 x[2,4,0] x[2,4,1] x[2,4,2] x[2,5,0] x[2,5,1] x[2,5,2] x[2,6,0] x[2,6,1]
 x[2,6,2] x[2,7,0] x[2,7,1] x[2,7,2] x[2,8,0] x[2,8,1] x[2,8,2] x[3,4,0]
 x[3,4,1] x[3,4,2] x[3,5,0] x[3,5,1] x[3,5,2] x[3,6,0] x[3,6,1] x[3,6,2]
 x[3,7,0] x[3,7,1] x[3,7,2] x[3,8,0] x[3,8,1] x[3,8,2] x[4,5,0] x[4,5,1]
 x[4,5,2] x[4,6,0] x[4,6,1] x[4,6,2] x[4,7,0] x[4,7,1] x[4,7,2] x[4,8,0]
 x[4,8,1] x[4,8,2] x[5,4,0] x[5,4,1] x[5,4,2] x[5,6,0] x[5,6,1] x[5,6,2]
 x[5,7,0] x[5,7,1] x[5,7,2] x[5,8,0] x[5,8,1] x[5,8,2] x[6,4,0] x[6,4,1]
 x[6,4,2] x[6,5,0] x[6,5,1] x[6,5,2] x[6,7,0] x[6,7,1] x[6,7,2] x[6,8,0]
 x[6,8,1] x[6,8,2] x[7,4,0] x[7,4,1] x[7,4,2] x[7,5,0] x[7,5,1] x[7,5,2]
 x[7,6,0] x[7,6,1] x[7,6,2] x[7,8,0] x[7,8,1] x[7,8,2] x[8,4,0] x[8,4,1]
 x[8,4,2] x[8,5,0] x[8,5,1] x[8,5,2] x[8,6,0] x[8,6,1] x[8,6,2] x[8,7,0]
 x[8,7,1] x[8,7,2] x[4,0,0] x[4,0,1] x[4,0,2] x[5,0,0] x[5,0,1] x[5,0,2]
 x[6,0,0] x[6,0,1] x[6,0,2] x[7,0,0] x[7,0,1] x[7,0,2] x[8,0,0] x[8,0,1]
 x[8,0,2] x_k[0] x_k[1] x_k[2] y_1[0,1] y_2[0,1] y_1[0,2] y_2[0,2] y_1[1,2]
 y_2[1,2]
End
