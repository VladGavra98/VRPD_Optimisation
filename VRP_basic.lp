\ Model VRP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  OBJ0: Priority=2 Weight=1 AbsTol=1e-06 RelTol=0
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
   + 3326 x[8,0,2] + 0 y_1[0,1] + 0 y_2[0,1] + 0 y_1[0,2] + 0 y_2[0,2]
   + 0 y_1[1,2] + 0 y_2[1,2]
  OBJ1: Priority=1 Weight=1 AbsTol=1e-06 RelTol=0
   - launch[0] - launch[1] - launch[2] + land[0] + land[1] + land[2]
  OBJ2: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
   tau[1,0] + tau[2,0] + tau[3,0] + tau[4,0] + tau[5,0] + tau[6,0] + tau[7,0]
   + tau[8,0] + tau[1,1] + tau[2,1] + tau[3,1] + tau[4,1] + tau[5,1]
   + tau[6,1] + tau[7,1] + tau[8,1] + tau[1,2] + tau[2,2] + tau[3,2]
   + tau[4,2] + tau[5,2] + tau[6,2] + tau[7,2] + tau[8,2]
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
 time_bound_on_pizzeria[1,0]: 100000 x[0,1,0] - tau[1,0] <= 99400
 time_bound_on_pizzeria[1,1]: 100000 x[0,1,1] - tau[1,1] <= 99400
 time_bound_on_pizzeria[1,2]: 100000 x[0,1,2] - tau[1,2] <= 99400
 time_bound_on_pizzeria[2,0]: 100000 x[0,2,0] - tau[2,0] <= 99300
 time_bound_on_pizzeria[2,1]: 100000 x[0,2,1] - tau[2,1] <= 99300
 time_bound_on_pizzeria[2,2]: 100000 x[0,2,2] - tau[2,2] <= 99300
 time_bound_on_pizzeria[3,0]: 100000 x[0,3,0] - tau[3,0] <= 99200
 time_bound_on_pizzeria[3,1]: 100000 x[0,3,1] - tau[3,1] <= 99200
 time_bound_on_pizzeria[3,2]: 100000 x[0,3,2] - tau[3,2] <= 99200
 Time_window_pizzeria[1,0]: 100000 x[0,1,0] - tau[1,0] <= 99836
 Time_window_pizzeria[1,1]: 100000 x[0,1,1] - tau[1,1] <= 99836
 Time_window_pizzeria[1,2]: 100000 x[0,1,2] - tau[1,2] <= 99836
 Time_window_pizzeria[2,0]: 100000 x[0,2,0] - tau[2,0] <= 99779.2
 Time_window_pizzeria[2,1]: 100000 x[0,2,1] - tau[2,1] <= 99779.2
 Time_window_pizzeria[2,2]: 100000 x[0,2,2] - tau[2,2] <= 99779.2
 Time_window_pizzeria[3,0]: 100000 x[0,3,0] - tau[3,0] <= 99752.8
 Time_window_pizzeria[3,1]: 100000 x[0,3,1] - tau[3,1] <= 99752.8
 Time_window_pizzeria[3,2]: 100000 x[0,3,2] - tau[3,2] <= 99752.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,4,0]:
   100000 x[1,4,0] + tau[1,0] - tau[4,0] <= 99670
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,4,1]:
   100000 x[1,4,1] + tau[1,1] - tau[4,1] <= 99670
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,4,2]:
   100000 x[1,4,2] + tau[1,2] - tau[4,2] <= 99670
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,5,0]:
   100000 x[1,5,0] + tau[1,0] - tau[5,0] <= 99680.5
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,5,1]:
   100000 x[1,5,1] + tau[1,1] - tau[5,1] <= 99680.5
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,5,2]:
   100000 x[1,5,2] + tau[1,2] - tau[5,2] <= 99680.5
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,6,0]:
   100000 x[1,6,0] + tau[1,0] - tau[6,0] <= 99715.7
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,6,1]:
   100000 x[1,6,1] + tau[1,1] - tau[6,1] <= 99715.7
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,6,2]:
   100000 x[1,6,2] + tau[1,2] - tau[6,2] <= 99715.7
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,7,0]:
   100000 x[1,7,0] + tau[1,0] - tau[7,0] <= 99744.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,7,1]:
   100000 x[1,7,1] + tau[1,1] - tau[7,1] <= 99744.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,7,2]:
   100000 x[1,7,2] + tau[1,2] - tau[7,2] <= 99744.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,8,0]:
   100000 x[1,8,0] + tau[1,0] - tau[8,0] <= 99824.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,8,1]:
   100000 x[1,8,1] + tau[1,1] - tau[8,1] <= 99824.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,8,2]:
   100000 x[1,8,2] + tau[1,2] - tau[8,2] <= 99824.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,4,0]:
   100000 x[2,4,0] + tau[2,0] - tau[4,0] <= 99692.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,4,1]:
   100000 x[2,4,1] + tau[2,1] - tau[4,1] <= 99692.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,4,2]:
   100000 x[2,4,2] + tau[2,2] - tau[4,2] <= 99692.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,5,0]:
   100000 x[2,5,0] + tau[2,0] - tau[5,0] <= 99705.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,5,1]:
   100000 x[2,5,1] + tau[2,1] - tau[5,1] <= 99705.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,5,2]:
   100000 x[2,5,2] + tau[2,2] - tau[5,2] <= 99705.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,6,0]:
   100000 x[2,6,0] + tau[2,0] - tau[6,0] <= 99734.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,6,1]:
   100000 x[2,6,1] + tau[2,1] - tau[6,1] <= 99734.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,6,2]:
   100000 x[2,6,2] + tau[2,2] - tau[6,2] <= 99734.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,7,0]:
   100000 x[2,7,0] + tau[2,0] - tau[7,0] <= 99724.5
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,7,1]:
   100000 x[2,7,1] + tau[2,1] - tau[7,1] <= 99724.5
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,7,2]:
   100000 x[2,7,2] + tau[2,2] - tau[7,2] <= 99724.5
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,8,0]:
   100000 x[2,8,0] + tau[2,0] - tau[8,0] <= 99888.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,8,1]:
   100000 x[2,8,1] + tau[2,1] - tau[8,1] <= 99888.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,8,2]:
   100000 x[2,8,2] + tau[2,2] - tau[8,2] <= 99888.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,4,0]:
   100000 x[3,4,0] + tau[3,0] - tau[4,0] <= 99693.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,4,1]:
   100000 x[3,4,1] + tau[3,1] - tau[4,1] <= 99693.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,4,2]:
   100000 x[3,4,2] + tau[3,2] - tau[4,2] <= 99693.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,5,0]:
   100000 x[3,5,0] + tau[3,0] - tau[5,0] <= 99707.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,5,1]:
   100000 x[3,5,1] + tau[3,1] - tau[5,1] <= 99707.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,5,2]:
   100000 x[3,5,2] + tau[3,2] - tau[5,2] <= 99707.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,6,0]:
   100000 x[3,6,0] + tau[3,0] - tau[6,0] <= 99733.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,6,1]:
   100000 x[3,6,1] + tau[3,1] - tau[6,1] <= 99733.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,6,2]:
   100000 x[3,6,2] + tau[3,2] - tau[6,2] <= 99733.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,7,0]:
   100000 x[3,7,0] + tau[3,0] - tau[7,0] <= 99706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,7,1]:
   100000 x[3,7,1] + tau[3,1] - tau[7,1] <= 99706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,7,2]:
   100000 x[3,7,2] + tau[3,2] - tau[7,2] <= 99706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,8,0]:
   100000 x[3,8,0] + tau[3,0] - tau[8,0] <= 99913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,8,1]:
   100000 x[3,8,1] + tau[3,1] - tau[8,1] <= 99913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,8,2]:
   100000 x[3,8,2] + tau[3,2] - tau[8,2] <= 99913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,5,0]:
   100000 x[4,5,0] + tau[4,0] - tau[5,0] <= 99983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,5,1]:
   100000 x[4,5,1] + tau[4,1] - tau[5,1] <= 99983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,5,2]:
   100000 x[4,5,2] + tau[4,2] - tau[5,2] <= 99983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,6,0]:
   100000 x[4,6,0] + tau[4,0] - tau[6,0] <= 99952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,6,1]:
   100000 x[4,6,1] + tau[4,1] - tau[6,1] <= 99952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,6,2]:
   100000 x[4,6,2] + tau[4,2] - tau[6,2] <= 99952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,7,0]:
   100000 x[4,7,0] + tau[4,0] - tau[7,0] <= 99810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,7,1]:
   100000 x[4,7,1] + tau[4,1] - tau[7,1] <= 99810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,7,2]:
   100000 x[4,7,2] + tau[4,2] - tau[7,2] <= 99810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,8,0]:
   100000 x[4,8,0] + tau[4,0] - tau[8,0] <= 99743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,8,1]:
   100000 x[4,8,1] + tau[4,1] - tau[8,1] <= 99743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,8,2]:
   100000 x[4,8,2] + tau[4,2] - tau[8,2] <= 99743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,4,0]:
   100000 x[5,4,0] - tau[4,0] + tau[5,0] <= 99983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,4,1]:
   100000 x[5,4,1] - tau[4,1] + tau[5,1] <= 99983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,4,2]:
   100000 x[5,4,2] - tau[4,2] + tau[5,2] <= 99983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,6,0]:
   100000 x[5,6,0] + tau[5,0] - tau[6,0] <= 99957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,6,1]:
   100000 x[5,6,1] + tau[5,1] - tau[6,1] <= 99957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,6,2]:
   100000 x[5,6,2] + tau[5,2] - tau[6,2] <= 99957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,7,0]:
   100000 x[5,7,0] + tau[5,0] - tau[7,0] <= 99806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,7,1]:
   100000 x[5,7,1] + tau[5,1] - tau[7,1] <= 99806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,7,2]:
   100000 x[5,7,2] + tau[5,2] - tau[7,2] <= 99806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,8,0]:
   100000 x[5,8,0] + tau[5,0] - tau[8,0] <= 99759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,8,1]:
   100000 x[5,8,1] + tau[5,1] - tau[8,1] <= 99759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,8,2]:
   100000 x[5,8,2] + tau[5,2] - tau[8,2] <= 99759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,4,0]:
   100000 x[6,4,0] - tau[4,0] + tau[6,0] <= 99952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,4,1]:
   100000 x[6,4,1] - tau[4,1] + tau[6,1] <= 99952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,4,2]:
   100000 x[6,4,2] - tau[4,2] + tau[6,2] <= 99952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,5,0]:
   100000 x[6,5,0] - tau[5,0] + tau[6,0] <= 99957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,5,1]:
   100000 x[6,5,1] - tau[5,1] + tau[6,1] <= 99957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,5,2]:
   100000 x[6,5,2] - tau[5,2] + tau[6,2] <= 99957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,7,0]:
   100000 x[6,7,0] + tau[6,0] - tau[7,0] <= 99845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,7,1]:
   100000 x[6,7,1] + tau[6,1] - tau[7,1] <= 99845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,7,2]:
   100000 x[6,7,2] + tau[6,2] - tau[7,2] <= 99845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,8,0]:
   100000 x[6,8,0] + tau[6,0] - tau[8,0] <= 99773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,8,1]:
   100000 x[6,8,1] + tau[6,1] - tau[8,1] <= 99773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,8,2]:
   100000 x[6,8,2] + tau[6,2] - tau[8,2] <= 99773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,4,0]:
   100000 x[7,4,0] - tau[4,0] + tau[7,0] <= 99810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,4,1]:
   100000 x[7,4,1] - tau[4,1] + tau[7,1] <= 99810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,4,2]:
   100000 x[7,4,2] - tau[4,2] + tau[7,2] <= 99810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,5,0]:
   100000 x[7,5,0] - tau[5,0] + tau[7,0] <= 99806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,5,1]:
   100000 x[7,5,1] - tau[5,1] + tau[7,1] <= 99806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,5,2]:
   100000 x[7,5,2] - tau[5,2] + tau[7,2] <= 99806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,6,0]:
   100000 x[7,6,0] - tau[6,0] + tau[7,0] <= 99845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,6,1]:
   100000 x[7,6,1] - tau[6,1] + tau[7,1] <= 99845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,6,2]:
   100000 x[7,6,2] - tau[6,2] + tau[7,2] <= 99845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,8,0]:
   100000 x[7,8,0] + tau[7,0] - tau[8,0] <= 99701.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,8,1]:
   100000 x[7,8,1] + tau[7,1] - tau[8,1] <= 99701.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,8,2]:
   100000 x[7,8,2] + tau[7,2] - tau[8,2] <= 99701.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,4,0]:
   100000 x[8,4,0] - tau[4,0] + tau[8,0] <= 99743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,4,1]:
   100000 x[8,4,1] - tau[4,1] + tau[8,1] <= 99743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,4,2]:
   100000 x[8,4,2] - tau[4,2] + tau[8,2] <= 99743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,5,0]:
   100000 x[8,5,0] - tau[5,0] + tau[8,0] <= 99759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,5,1]:
   100000 x[8,5,1] - tau[5,1] + tau[8,1] <= 99759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,5,2]:
   100000 x[8,5,2] - tau[5,2] + tau[8,2] <= 99759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,6,0]:
   100000 x[8,6,0] - tau[6,0] + tau[8,0] <= 99773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,6,1]:
   100000 x[8,6,1] - tau[6,1] + tau[8,1] <= 99773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,6,2]:
   100000 x[8,6,2] - tau[6,2] + tau[8,2] <= 99773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,7,0]:
   100000 x[8,7,0] - tau[7,0] + tau[8,0] <= 99701.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,7,1]:
   100000 x[8,7,1] - tau[7,1] + tau[8,1] <= 99701.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,7,2]:
   100000 x[8,7,2] - tau[7,2] + tau[8,2] <= 99701.3
 lower_bound_on_customer_[4,1,0]: 100000 x[1,4,0] - tau[4,0] <= 99300
 lower_bound_on_customer_[4,1,1]: 100000 x[1,4,1] - tau[4,1] <= 99300
 lower_bound_on_customer_[4,1,2]: 100000 x[1,4,2] - tau[4,2] <= 99300
 lower_bound_on_customer_[4,2,0]: 100000 x[2,4,0] - tau[4,0] <= 99300
 lower_bound_on_customer_[4,2,1]: 100000 x[2,4,1] - tau[4,1] <= 99300
 lower_bound_on_customer_[4,2,2]: 100000 x[2,4,2] - tau[4,2] <= 99300
 lower_bound_on_customer_[4,3,0]: 100000 x[3,4,0] - tau[4,0] <= 99300
 lower_bound_on_customer_[4,3,1]: 100000 x[3,4,1] - tau[4,1] <= 99300
 lower_bound_on_customer_[4,3,2]: 100000 x[3,4,2] - tau[4,2] <= 99300
 lower_bound_on_customer_[4,5,0]: 100000 x[5,4,0] - tau[4,0] <= 99300
 lower_bound_on_customer_[4,5,1]: 100000 x[5,4,1] - tau[4,1] <= 99300
 lower_bound_on_customer_[4,5,2]: 100000 x[5,4,2] - tau[4,2] <= 99300
 lower_bound_on_customer_[4,6,0]: 100000 x[6,4,0] - tau[4,0] <= 99300
 lower_bound_on_customer_[4,6,1]: 100000 x[6,4,1] - tau[4,1] <= 99300
 lower_bound_on_customer_[4,6,2]: 100000 x[6,4,2] - tau[4,2] <= 99300
 lower_bound_on_customer_[4,7,0]: 100000 x[7,4,0] - tau[4,0] <= 99300
 lower_bound_on_customer_[4,7,1]: 100000 x[7,4,1] - tau[4,1] <= 99300
 lower_bound_on_customer_[4,7,2]: 100000 x[7,4,2] - tau[4,2] <= 99300
 lower_bound_on_customer_[4,8,0]: 100000 x[8,4,0] - tau[4,0] <= 99300
 lower_bound_on_customer_[4,8,1]: 100000 x[8,4,1] - tau[4,1] <= 99300
 lower_bound_on_customer_[4,8,2]: 100000 x[8,4,2] - tau[4,2] <= 99300
 lower_bound_on_customer_[5,1,0]: 100000 x[1,5,0] - tau[5,0] <= 99300
 lower_bound_on_customer_[5,1,1]: 100000 x[1,5,1] - tau[5,1] <= 99300
 lower_bound_on_customer_[5,1,2]: 100000 x[1,5,2] - tau[5,2] <= 99300
 lower_bound_on_customer_[5,2,0]: 100000 x[2,5,0] - tau[5,0] <= 99300
 lower_bound_on_customer_[5,2,1]: 100000 x[2,5,1] - tau[5,1] <= 99300
 lower_bound_on_customer_[5,2,2]: 100000 x[2,5,2] - tau[5,2] <= 99300
 lower_bound_on_customer_[5,3,0]: 100000 x[3,5,0] - tau[5,0] <= 99300
 lower_bound_on_customer_[5,3,1]: 100000 x[3,5,1] - tau[5,1] <= 99300
 lower_bound_on_customer_[5,3,2]: 100000 x[3,5,2] - tau[5,2] <= 99300
 lower_bound_on_customer_[5,4,0]: 100000 x[4,5,0] - tau[5,0] <= 99300
 lower_bound_on_customer_[5,4,1]: 100000 x[4,5,1] - tau[5,1] <= 99300
 lower_bound_on_customer_[5,4,2]: 100000 x[4,5,2] - tau[5,2] <= 99300
 lower_bound_on_customer_[5,6,0]: 100000 x[6,5,0] - tau[5,0] <= 99300
 lower_bound_on_customer_[5,6,1]: 100000 x[6,5,1] - tau[5,1] <= 99300
 lower_bound_on_customer_[5,6,2]: 100000 x[6,5,2] - tau[5,2] <= 99300
 lower_bound_on_customer_[5,7,0]: 100000 x[7,5,0] - tau[5,0] <= 99300
 lower_bound_on_customer_[5,7,1]: 100000 x[7,5,1] - tau[5,1] <= 99300
 lower_bound_on_customer_[5,7,2]: 100000 x[7,5,2] - tau[5,2] <= 99300
 lower_bound_on_customer_[5,8,0]: 100000 x[8,5,0] - tau[5,0] <= 99300
 lower_bound_on_customer_[5,8,1]: 100000 x[8,5,1] - tau[5,1] <= 99300
 lower_bound_on_customer_[5,8,2]: 100000 x[8,5,2] - tau[5,2] <= 99300
 lower_bound_on_customer_[6,1,0]: 100000 x[1,6,0] - tau[6,0] <= 99300
 lower_bound_on_customer_[6,1,1]: 100000 x[1,6,1] - tau[6,1] <= 99300
 lower_bound_on_customer_[6,1,2]: 100000 x[1,6,2] - tau[6,2] <= 99300
 lower_bound_on_customer_[6,2,0]: 100000 x[2,6,0] - tau[6,0] <= 99300
 lower_bound_on_customer_[6,2,1]: 100000 x[2,6,1] - tau[6,1] <= 99300
 lower_bound_on_customer_[6,2,2]: 100000 x[2,6,2] - tau[6,2] <= 99300
 lower_bound_on_customer_[6,3,0]: 100000 x[3,6,0] - tau[6,0] <= 99300
 lower_bound_on_customer_[6,3,1]: 100000 x[3,6,1] - tau[6,1] <= 99300
 lower_bound_on_customer_[6,3,2]: 100000 x[3,6,2] - tau[6,2] <= 99300
 lower_bound_on_customer_[6,4,0]: 100000 x[4,6,0] - tau[6,0] <= 99300
 lower_bound_on_customer_[6,4,1]: 100000 x[4,6,1] - tau[6,1] <= 99300
 lower_bound_on_customer_[6,4,2]: 100000 x[4,6,2] - tau[6,2] <= 99300
 lower_bound_on_customer_[6,5,0]: 100000 x[5,6,0] - tau[6,0] <= 99300
 lower_bound_on_customer_[6,5,1]: 100000 x[5,6,1] - tau[6,1] <= 99300
 lower_bound_on_customer_[6,5,2]: 100000 x[5,6,2] - tau[6,2] <= 99300
 lower_bound_on_customer_[6,7,0]: 100000 x[7,6,0] - tau[6,0] <= 99300
 lower_bound_on_customer_[6,7,1]: 100000 x[7,6,1] - tau[6,1] <= 99300
 lower_bound_on_customer_[6,7,2]: 100000 x[7,6,2] - tau[6,2] <= 99300
 lower_bound_on_customer_[6,8,0]: 100000 x[8,6,0] - tau[6,0] <= 99300
 lower_bound_on_customer_[6,8,1]: 100000 x[8,6,1] - tau[6,1] <= 99300
 lower_bound_on_customer_[6,8,2]: 100000 x[8,6,2] - tau[6,2] <= 99300
 lower_bound_on_customer_[7,1,0]: 100000 x[1,7,0] - tau[7,0] <= 99300
 lower_bound_on_customer_[7,1,1]: 100000 x[1,7,1] - tau[7,1] <= 99300
 lower_bound_on_customer_[7,1,2]: 100000 x[1,7,2] - tau[7,2] <= 99300
 lower_bound_on_customer_[7,2,0]: 100000 x[2,7,0] - tau[7,0] <= 99300
 lower_bound_on_customer_[7,2,1]: 100000 x[2,7,1] - tau[7,1] <= 99300
 lower_bound_on_customer_[7,2,2]: 100000 x[2,7,2] - tau[7,2] <= 99300
 lower_bound_on_customer_[7,3,0]: 100000 x[3,7,0] - tau[7,0] <= 99300
 lower_bound_on_customer_[7,3,1]: 100000 x[3,7,1] - tau[7,1] <= 99300
 lower_bound_on_customer_[7,3,2]: 100000 x[3,7,2] - tau[7,2] <= 99300
 lower_bound_on_customer_[7,4,0]: 100000 x[4,7,0] - tau[7,0] <= 99300
 lower_bound_on_customer_[7,4,1]: 100000 x[4,7,1] - tau[7,1] <= 99300
 lower_bound_on_customer_[7,4,2]: 100000 x[4,7,2] - tau[7,2] <= 99300
 lower_bound_on_customer_[7,5,0]: 100000 x[5,7,0] - tau[7,0] <= 99300
 lower_bound_on_customer_[7,5,1]: 100000 x[5,7,1] - tau[7,1] <= 99300
 lower_bound_on_customer_[7,5,2]: 100000 x[5,7,2] - tau[7,2] <= 99300
 lower_bound_on_customer_[7,6,0]: 100000 x[6,7,0] - tau[7,0] <= 99300
 lower_bound_on_customer_[7,6,1]: 100000 x[6,7,1] - tau[7,1] <= 99300
 lower_bound_on_customer_[7,6,2]: 100000 x[6,7,2] - tau[7,2] <= 99300
 lower_bound_on_customer_[7,8,0]: 100000 x[8,7,0] - tau[7,0] <= 99300
 lower_bound_on_customer_[7,8,1]: 100000 x[8,7,1] - tau[7,1] <= 99300
 lower_bound_on_customer_[7,8,2]: 100000 x[8,7,2] - tau[7,2] <= 99300
 lower_bound_on_customer_[8,1,0]: 100000 x[1,8,0] - tau[8,0] <= 99300
 lower_bound_on_customer_[8,1,1]: 100000 x[1,8,1] - tau[8,1] <= 99300
 lower_bound_on_customer_[8,1,2]: 100000 x[1,8,2] - tau[8,2] <= 99300
 lower_bound_on_customer_[8,2,0]: 100000 x[2,8,0] - tau[8,0] <= 99300
 lower_bound_on_customer_[8,2,1]: 100000 x[2,8,1] - tau[8,1] <= 99300
 lower_bound_on_customer_[8,2,2]: 100000 x[2,8,2] - tau[8,2] <= 99300
 lower_bound_on_customer_[8,3,0]: 100000 x[3,8,0] - tau[8,0] <= 99300
 lower_bound_on_customer_[8,3,1]: 100000 x[3,8,1] - tau[8,1] <= 99300
 lower_bound_on_customer_[8,3,2]: 100000 x[3,8,2] - tau[8,2] <= 99300
 lower_bound_on_customer_[8,4,0]: 100000 x[4,8,0] - tau[8,0] <= 99300
 lower_bound_on_customer_[8,4,1]: 100000 x[4,8,1] - tau[8,1] <= 99300
 lower_bound_on_customer_[8,4,2]: 100000 x[4,8,2] - tau[8,2] <= 99300
 lower_bound_on_customer_[8,5,0]: 100000 x[5,8,0] - tau[8,0] <= 99300
 lower_bound_on_customer_[8,5,1]: 100000 x[5,8,1] - tau[8,1] <= 99300
 lower_bound_on_customer_[8,5,2]: 100000 x[5,8,2] - tau[8,2] <= 99300
 lower_bound_on_customer_[8,6,0]: 100000 x[6,8,0] - tau[8,0] <= 99300
 lower_bound_on_customer_[8,6,1]: 100000 x[6,8,1] - tau[8,1] <= 99300
 lower_bound_on_customer_[8,6,2]: 100000 x[6,8,2] - tau[8,2] <= 99300
 lower_bound_on_customer_[8,7,0]: 100000 x[7,8,0] - tau[8,0] <= 99300
 lower_bound_on_customer_[8,7,1]: 100000 x[7,8,1] - tau[8,1] <= 99300
 lower_bound_on_customer_[8,7,2]: 100000 x[7,8,2] - tau[8,2] <= 99300
 upper_bound_on_customer_[4,1,0]: - 100000 x[1,4,0] - tau[4,0] >= -101300
 upper_bound_on_customer_[4,1,1]: - 100000 x[1,4,1] - tau[4,1] >= -101300
 upper_bound_on_customer_[4,1,2]: - 100000 x[1,4,2] - tau[4,2] >= -101300
 upper_bound_on_customer_[4,2,0]: - 100000 x[2,4,0] - tau[4,0] >= -101300
 upper_bound_on_customer_[4,2,1]: - 100000 x[2,4,1] - tau[4,1] >= -101300
 upper_bound_on_customer_[4,2,2]: - 100000 x[2,4,2] - tau[4,2] >= -101300
 upper_bound_on_customer_[4,3,0]: - 100000 x[3,4,0] - tau[4,0] >= -101300
 upper_bound_on_customer_[4,3,1]: - 100000 x[3,4,1] - tau[4,1] >= -101300
 upper_bound_on_customer_[4,3,2]: - 100000 x[3,4,2] - tau[4,2] >= -101300
 upper_bound_on_customer_[4,5,0]: - 100000 x[5,4,0] - tau[4,0] >= -101300
 upper_bound_on_customer_[4,5,1]: - 100000 x[5,4,1] - tau[4,1] >= -101300
 upper_bound_on_customer_[4,5,2]: - 100000 x[5,4,2] - tau[4,2] >= -101300
 upper_bound_on_customer_[4,6,0]: - 100000 x[6,4,0] - tau[4,0] >= -101300
 upper_bound_on_customer_[4,6,1]: - 100000 x[6,4,1] - tau[4,1] >= -101300
 upper_bound_on_customer_[4,6,2]: - 100000 x[6,4,2] - tau[4,2] >= -101300
 upper_bound_on_customer_[4,7,0]: - 100000 x[7,4,0] - tau[4,0] >= -101300
 upper_bound_on_customer_[4,7,1]: - 100000 x[7,4,1] - tau[4,1] >= -101300
 upper_bound_on_customer_[4,7,2]: - 100000 x[7,4,2] - tau[4,2] >= -101300
 upper_bound_on_customer_[4,8,0]: - 100000 x[8,4,0] - tau[4,0] >= -101300
 upper_bound_on_customer_[4,8,1]: - 100000 x[8,4,1] - tau[4,1] >= -101300
 upper_bound_on_customer_[4,8,2]: - 100000 x[8,4,2] - tau[4,2] >= -101300
 upper_bound_on_customer_[5,1,0]: - 100000 x[1,5,0] - tau[5,0] >= -101400
 upper_bound_on_customer_[5,1,1]: - 100000 x[1,5,1] - tau[5,1] >= -101400
 upper_bound_on_customer_[5,1,2]: - 100000 x[1,5,2] - tau[5,2] >= -101400
 upper_bound_on_customer_[5,2,0]: - 100000 x[2,5,0] - tau[5,0] >= -101400
 upper_bound_on_customer_[5,2,1]: - 100000 x[2,5,1] - tau[5,1] >= -101400
 upper_bound_on_customer_[5,2,2]: - 100000 x[2,5,2] - tau[5,2] >= -101400
 upper_bound_on_customer_[5,3,0]: - 100000 x[3,5,0] - tau[5,0] >= -101400
 upper_bound_on_customer_[5,3,1]: - 100000 x[3,5,1] - tau[5,1] >= -101400
 upper_bound_on_customer_[5,3,2]: - 100000 x[3,5,2] - tau[5,2] >= -101400
 upper_bound_on_customer_[5,4,0]: - 100000 x[4,5,0] - tau[5,0] >= -101400
 upper_bound_on_customer_[5,4,1]: - 100000 x[4,5,1] - tau[5,1] >= -101400
 upper_bound_on_customer_[5,4,2]: - 100000 x[4,5,2] - tau[5,2] >= -101400
 upper_bound_on_customer_[5,6,0]: - 100000 x[6,5,0] - tau[5,0] >= -101400
 upper_bound_on_customer_[5,6,1]: - 100000 x[6,5,1] - tau[5,1] >= -101400
 upper_bound_on_customer_[5,6,2]: - 100000 x[6,5,2] - tau[5,2] >= -101400
 upper_bound_on_customer_[5,7,0]: - 100000 x[7,5,0] - tau[5,0] >= -101400
 upper_bound_on_customer_[5,7,1]: - 100000 x[7,5,1] - tau[5,1] >= -101400
 upper_bound_on_customer_[5,7,2]: - 100000 x[7,5,2] - tau[5,2] >= -101400
 upper_bound_on_customer_[5,8,0]: - 100000 x[8,5,0] - tau[5,0] >= -101400
 upper_bound_on_customer_[5,8,1]: - 100000 x[8,5,1] - tau[5,1] >= -101400
 upper_bound_on_customer_[5,8,2]: - 100000 x[8,5,2] - tau[5,2] >= -101400
 upper_bound_on_customer_[6,1,0]: - 100000 x[1,6,0] - tau[6,0] >= -101500
 upper_bound_on_customer_[6,1,1]: - 100000 x[1,6,1] - tau[6,1] >= -101500
 upper_bound_on_customer_[6,1,2]: - 100000 x[1,6,2] - tau[6,2] >= -101500
 upper_bound_on_customer_[6,2,0]: - 100000 x[2,6,0] - tau[6,0] >= -101500
 upper_bound_on_customer_[6,2,1]: - 100000 x[2,6,1] - tau[6,1] >= -101500
 upper_bound_on_customer_[6,2,2]: - 100000 x[2,6,2] - tau[6,2] >= -101500
 upper_bound_on_customer_[6,3,0]: - 100000 x[3,6,0] - tau[6,0] >= -101500
 upper_bound_on_customer_[6,3,1]: - 100000 x[3,6,1] - tau[6,1] >= -101500
 upper_bound_on_customer_[6,3,2]: - 100000 x[3,6,2] - tau[6,2] >= -101500
 upper_bound_on_customer_[6,4,0]: - 100000 x[4,6,0] - tau[6,0] >= -101500
 upper_bound_on_customer_[6,4,1]: - 100000 x[4,6,1] - tau[6,1] >= -101500
 upper_bound_on_customer_[6,4,2]: - 100000 x[4,6,2] - tau[6,2] >= -101500
 upper_bound_on_customer_[6,5,0]: - 100000 x[5,6,0] - tau[6,0] >= -101500
 upper_bound_on_customer_[6,5,1]: - 100000 x[5,6,1] - tau[6,1] >= -101500
 upper_bound_on_customer_[6,5,2]: - 100000 x[5,6,2] - tau[6,2] >= -101500
 upper_bound_on_customer_[6,7,0]: - 100000 x[7,6,0] - tau[6,0] >= -101500
 upper_bound_on_customer_[6,7,1]: - 100000 x[7,6,1] - tau[6,1] >= -101500
 upper_bound_on_customer_[6,7,2]: - 100000 x[7,6,2] - tau[6,2] >= -101500
 upper_bound_on_customer_[6,8,0]: - 100000 x[8,6,0] - tau[6,0] >= -101500
 upper_bound_on_customer_[6,8,1]: - 100000 x[8,6,1] - tau[6,1] >= -101500
 upper_bound_on_customer_[6,8,2]: - 100000 x[8,6,2] - tau[6,2] >= -101500
 upper_bound_on_customer_[7,1,0]: - 100000 x[1,7,0] - tau[7,0] >= -101600
 upper_bound_on_customer_[7,1,1]: - 100000 x[1,7,1] - tau[7,1] >= -101600
 upper_bound_on_customer_[7,1,2]: - 100000 x[1,7,2] - tau[7,2] >= -101600
 upper_bound_on_customer_[7,2,0]: - 100000 x[2,7,0] - tau[7,0] >= -101600
 upper_bound_on_customer_[7,2,1]: - 100000 x[2,7,1] - tau[7,1] >= -101600
 upper_bound_on_customer_[7,2,2]: - 100000 x[2,7,2] - tau[7,2] >= -101600
 upper_bound_on_customer_[7,3,0]: - 100000 x[3,7,0] - tau[7,0] >= -101600
 upper_bound_on_customer_[7,3,1]: - 100000 x[3,7,1] - tau[7,1] >= -101600
 upper_bound_on_customer_[7,3,2]: - 100000 x[3,7,2] - tau[7,2] >= -101600
 upper_bound_on_customer_[7,4,0]: - 100000 x[4,7,0] - tau[7,0] >= -101600
 upper_bound_on_customer_[7,4,1]: - 100000 x[4,7,1] - tau[7,1] >= -101600
 upper_bound_on_customer_[7,4,2]: - 100000 x[4,7,2] - tau[7,2] >= -101600
 upper_bound_on_customer_[7,5,0]: - 100000 x[5,7,0] - tau[7,0] >= -101600
 upper_bound_on_customer_[7,5,1]: - 100000 x[5,7,1] - tau[7,1] >= -101600
 upper_bound_on_customer_[7,5,2]: - 100000 x[5,7,2] - tau[7,2] >= -101600
 upper_bound_on_customer_[7,6,0]: - 100000 x[6,7,0] - tau[7,0] >= -101600
 upper_bound_on_customer_[7,6,1]: - 100000 x[6,7,1] - tau[7,1] >= -101600
 upper_bound_on_customer_[7,6,2]: - 100000 x[6,7,2] - tau[7,2] >= -101600
 upper_bound_on_customer_[7,8,0]: - 100000 x[8,7,0] - tau[7,0] >= -101600
 upper_bound_on_customer_[7,8,1]: - 100000 x[8,7,1] - tau[7,1] >= -101600
 upper_bound_on_customer_[7,8,2]: - 100000 x[8,7,2] - tau[7,2] >= -101600
 upper_bound_on_customer_[8,1,0]: - 100000 x[1,8,0] - tau[8,0] >= -101700
 upper_bound_on_customer_[8,1,1]: - 100000 x[1,8,1] - tau[8,1] >= -101700
 upper_bound_on_customer_[8,1,2]: - 100000 x[1,8,2] - tau[8,2] >= -101700
 upper_bound_on_customer_[8,2,0]: - 100000 x[2,8,0] - tau[8,0] >= -101700
 upper_bound_on_customer_[8,2,1]: - 100000 x[2,8,1] - tau[8,1] >= -101700
 upper_bound_on_customer_[8,2,2]: - 100000 x[2,8,2] - tau[8,2] >= -101700
 upper_bound_on_customer_[8,3,0]: - 100000 x[3,8,0] - tau[8,0] >= -101700
 upper_bound_on_customer_[8,3,1]: - 100000 x[3,8,1] - tau[8,1] >= -101700
 upper_bound_on_customer_[8,3,2]: - 100000 x[3,8,2] - tau[8,2] >= -101700
 upper_bound_on_customer_[8,4,0]: - 100000 x[4,8,0] - tau[8,0] >= -101700
 upper_bound_on_customer_[8,4,1]: - 100000 x[4,8,1] - tau[8,1] >= -101700
 upper_bound_on_customer_[8,4,2]: - 100000 x[4,8,2] - tau[8,2] >= -101700
 upper_bound_on_customer_[8,5,0]: - 100000 x[5,8,0] - tau[8,0] >= -101700
 upper_bound_on_customer_[8,5,1]: - 100000 x[5,8,1] - tau[8,1] >= -101700
 upper_bound_on_customer_[8,5,2]: - 100000 x[5,8,2] - tau[8,2] >= -101700
 upper_bound_on_customer_[8,6,0]: - 100000 x[6,8,0] - tau[8,0] >= -101700
 upper_bound_on_customer_[8,6,1]: - 100000 x[6,8,1] - tau[8,1] >= -101700
 upper_bound_on_customer_[8,6,2]: - 100000 x[6,8,2] - tau[8,2] >= -101700
 upper_bound_on_customer_[8,7,0]: - 100000 x[7,8,0] - tau[8,0] >= -101700
 upper_bound_on_customer_[8,7,1]: - 100000 x[7,8,1] - tau[8,1] >= -101700
 upper_bound_on_customer_[8,7,2]: - 100000 x[7,8,2] - tau[8,2] >= -101700
 max_endurance_of_drone[0]: - launch[0] + land[0] <= 1800
 max_endurance_of_drone[1]: - launch[1] + land[1] <= 1800
 max_endurance_of_drone[2]: - launch[2] + land[2] <= 1800
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
 launch_time[0]: - 164 x[0,1,0] - 220.8 x[0,2,0] - 247.2 x[0,3,0]
   - launch[0] + [ x[0,1,0] * tau[1,0] + x[0,2,0] * tau[2,0]
   + x[0,3,0] * tau[3,0] ] = 0
 launch_time[1]: - 164 x[0,1,1] - 220.8 x[0,2,1] - 247.2 x[0,3,1]
   - launch[1] + [ x[0,1,1] * tau[1,1] + x[0,2,1] * tau[2,1]
   + x[0,3,1] * tau[3,1] ] = 0
 launch_time[2]: - 164 x[0,1,2] - 220.8 x[0,2,2] - 247.2 x[0,3,2]
   - launch[2] + [ x[0,1,2] * tau[1,2] + x[0,2,2] * tau[2,2]
   + x[0,3,2] * tau[3,2] ] = 0
 land_time[0]: 482.2 x[4,0,0] + 473.7 x[5,0,0] + 435.1 x[6,0,0]
   + 363.4 x[7,0,0] + 332.6 x[8,0,0] - land[0] + [ x[4,0,0] * tau[4,0]
   + x[5,0,0] * tau[5,0] + x[6,0,0] * tau[6,0] + x[7,0,0] * tau[7,0]
   + x[8,0,0] * tau[8,0] ] = 0
 land_time[1]: 482.2 x[4,0,1] + 473.7 x[5,0,1] + 435.1 x[6,0,1]
   + 363.4 x[7,0,1] + 332.6 x[8,0,1] - land[1] + [ x[4,0,1] * tau[4,1]
   + x[5,0,1] * tau[5,1] + x[6,0,1] * tau[6,1] + x[7,0,1] * tau[7,1]
   + x[8,0,1] * tau[8,1] ] = 0
 land_time[2]: 482.2 x[4,0,2] + 473.7 x[5,0,2] + 435.1 x[6,0,2]
   + 363.4 x[7,0,2] + 332.6 x[8,0,2] - land[2] + [ x[4,0,2] * tau[4,2]
   + x[5,0,2] * tau[5,2] + x[6,0,2] * tau[6,2] + x[7,0,2] * tau[7,2]
   + x[8,0,2] * tau[8,2] ] = 0
 either_launch_time_of_drone_m_is_D_time_after_launch_time_of_drone_k[0,1]:
   launch[0] - launch[1] - 100000 y_1[0,1] + [ 30 x_k[0] * x_k[1] ] <= 0
 either_launch_time_of_drone_m_is_D_time_after_launch_time_of_drone_k[0,2]:
   launch[0] - launch[2] - 100000 y_1[0,2] + [ 30 x_k[0] * x_k[2] ] <= 0
 either_launch_time_of_drone_m_is_D_time_after_launch_time_of_drone_k[1,2]:
   launch[1] - launch[2] - 100000 y_1[1,2] + [ 30 x_k[1] * x_k[2] ] <= 0
 or_launch_time_of_drone_k_is_D_time_after_launch_time_of_drone_m[0,1]:
   - launch[0] + launch[1] + 100000 y_1[0,1] + [ 30 x_k[0] * x_k[1] ]
   <= 100000
 or_launch_time_of_drone_k_is_D_time_after_launch_time_of_drone_m[0,2]:
   - launch[0] + launch[2] + 100000 y_1[0,2] + [ 30 x_k[0] * x_k[2] ]
   <= 100000
 or_launch_time_of_drone_k_is_D_time_after_launch_time_of_drone_m[1,2]:
   - launch[1] + launch[2] + 100000 y_1[1,2] + [ 30 x_k[1] * x_k[2] ]
   <= 100000
 either_land_time_of_drone_m_is_D_time_after_land_time_of_drone_k[0,1]:
   land[0] - land[1] - 100000 y_2[0,1] + [ 30 x_k[0] * x_k[1] ] <= 0
 either_land_time_of_drone_m_is_D_time_after_land_time_of_drone_k[0,2]:
   land[0] - land[2] - 100000 y_2[0,2] + [ 30 x_k[0] * x_k[2] ] <= 0
 either_land_time_of_drone_m_is_D_time_after_land_time_of_drone_k[1,2]:
   land[1] - land[2] - 100000 y_2[1,2] + [ 30 x_k[1] * x_k[2] ] <= 0
 or_land_time_of_drone_k_is_D_time_after_land_time_of_drone_m[0,1]:
   - land[0] + land[1] + 100000 y_2[0,1] + [ 30 x_k[0] * x_k[1] ]
   <= 100000
 or_land_time_of_drone_k_is_D_time_after_land_time_of_drone_m[0,2]:
   - land[0] + land[2] + 100000 y_2[0,2] + [ 30 x_k[0] * x_k[2] ]
   <= 100000
 or_land_time_of_drone_k_is_D_time_after_land_time_of_drone_m[1,2]:
   - land[1] + land[2] + 100000 y_2[1,2] + [ 30 x_k[1] * x_k[2] ]
   <= 100000
Bounds
 tau[1,0] <= 1700
 tau[2,0] <= 1700
 tau[3,0] <= 1700
 tau[4,0] <= 1700
 tau[5,0] <= 1700
 tau[6,0] <= 1700
 tau[7,0] <= 1700
 tau[8,0] <= 1700
 tau[1,1] <= 1700
 tau[2,1] <= 1700
 tau[3,1] <= 1700
 tau[4,1] <= 1700
 tau[5,1] <= 1700
 tau[6,1] <= 1700
 tau[7,1] <= 1700
 tau[8,1] <= 1700
 tau[1,2] <= 1700
 tau[2,2] <= 1700
 tau[3,2] <= 1700
 tau[4,2] <= 1700
 tau[5,2] <= 1700
 tau[6,2] <= 1700
 tau[7,2] <= 1700
 tau[8,2] <= 1700
 launch[0] <= 700
 launch[1] <= 700
 launch[2] <= 700
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
