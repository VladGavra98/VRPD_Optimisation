\ Model VRP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2936 x[1,4,0] + 2936 x[1,4,1] + 2936 x[1,4,2] + 2936 x[1,5,0]
   + 2936 x[1,5,1] + 2936 x[1,5,2] + 2936 x[1,6,0] + 2936 x[1,6,1]
   + 2936 x[1,6,2] + 2936 x[1,7,0] + 2936 x[1,7,1] + 2936 x[1,7,2]
   + 2936 x[1,8,0] + 2936 x[1,8,1] + 2936 x[1,8,2] + 866 x[2,4,0]
   + 866 x[2,4,1] + 866 x[2,4,2] + 866 x[2,5,0] + 866 x[2,5,1]
   + 866 x[2,5,2] + 866 x[2,6,0] + 866 x[2,6,1] + 866 x[2,6,2]
   + 866 x[2,7,0] + 866 x[2,7,1] + 866 x[2,7,2] + 866 x[2,8,0]
   + 866 x[2,8,1] + 866 x[2,8,2] + 3300 x[3,4,0] + 3300 x[3,4,1]
   + 3300 x[3,4,2] + 3300 x[3,5,0] + 3300 x[3,5,1] + 3300 x[3,5,2]
   + 3300 x[3,6,0] + 3300 x[3,6,1] + 3300 x[3,6,2] + 3300 x[3,7,0]
   + 3300 x[3,7,1] + 3300 x[3,7,2] + 3300 x[3,8,0] + 3300 x[3,8,1]
   + 3300 x[3,8,2] + 168 x[4,5,0] + 168 x[4,5,1] + 168 x[4,5,2]
   + 472 x[4,6,0] + 472 x[4,6,1] + 472 x[4,6,2] + 1898 x[4,7,0]
   + 1898 x[4,7,1] + 1898 x[4,7,2] + 2568 x[4,8,0] + 2568 x[4,8,1]
   + 2568 x[4,8,2] + 168 x[5,4,0] + 168 x[5,4,1] + 168 x[5,4,2]
   + 424 x[5,6,0] + 424 x[5,6,1] + 424 x[5,6,2] + 1936 x[5,7,0]
   + 1936 x[5,7,1] + 1936 x[5,7,2] + 2409 x[5,8,0] + 2409 x[5,8,1]
   + 2409 x[5,8,2] + 472 x[6,4,0] + 472 x[6,4,1] + 472 x[6,4,2]
   + 424 x[6,5,0] + 424 x[6,5,1] + 424 x[6,5,2] + 1548 x[6,7,0]
   + 1548 x[6,7,1] + 1548 x[6,7,2] + 2261 x[6,8,0] + 2261 x[6,8,1]
   + 2261 x[6,8,2] + 1898 x[7,4,0] + 1898 x[7,4,1] + 1898 x[7,4,2]
   + 1936 x[7,5,0] + 1936 x[7,5,1] + 1936 x[7,5,2] + 1548 x[7,6,0]
   + 1548 x[7,6,1] + 1548 x[7,6,2] + 2987 x[7,8,0] + 2987 x[7,8,1]
   + 2987 x[7,8,2] + 2568 x[8,4,0] + 2568 x[8,4,1] + 2568 x[8,4,2]
   + 2409 x[8,5,0] + 2409 x[8,5,1] + 2409 x[8,5,2] + 2261 x[8,6,0]
   + 2261 x[8,6,1] + 2261 x[8,6,2] + 2987 x[8,7,0] + 2987 x[8,7,1]
   + 2987 x[8,7,2] + 4822 x[4,0,0] + 4822 x[4,0,1] + 4822 x[4,0,2]
   + 4737 x[5,0,0] + 4737 x[5,0,1] + 4737 x[5,0,2] + 4351 x[6,0,0]
   + 4351 x[6,0,1] + 4351 x[6,0,2] + 3634 x[7,0,0] + 3634 x[7,0,1]
   + 3634 x[7,0,2] + 3326 x[8,0,0] + 3326 x[8,0,1] + 3326 x[8,0,2]
   + 1.000001e+12 tau + 1.000001e+12 tau + 1.000001e+12 tau + 1e+12 tau
   + 1e+12 tau + 1e+12 tau + 1e+12 tau + 1e+12 tau - 3.6e+09 Constant
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
 time_bound_on_pizzeria[1]: - tau <= -600
 time_bound_on_pizzeria[2]: - tau <= -1200
 time_bound_on_pizzeria[3]: - tau <= -1800
 Time_window_pizzeria[1,0]: 1e+06 x[0,1,0] - tau <= 999836
 Time_window_pizzeria[1,1]: 1e+06 x[0,1,1] - tau <= 999836
 Time_window_pizzeria[1,2]: 1e+06 x[0,1,2] - tau <= 999836
 Time_window_pizzeria[2,0]: 1e+06 x[0,2,0] - tau <= 999779.2
 Time_window_pizzeria[2,1]: 1e+06 x[0,2,1] - tau <= 999779.2
 Time_window_pizzeria[2,2]: 1e+06 x[0,2,2] - tau <= 999779.2
 Time_window_pizzeria[3,0]: 1e+06 x[0,3,0] - tau <= 999752.8
 Time_window_pizzeria[3,1]: 1e+06 x[0,3,1] - tau <= 999752.8
 Time_window_pizzeria[3,2]: 1e+06 x[0,3,2] - tau <= 999752.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,4,0]:
   1e+06 x[1,4,0] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,4,1]:
   1e+06 x[1,4,1] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,4,2]:
   1e+06 x[1,4,2] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,5,0]:
   1e+06 x[1,5,0] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,5,1]:
   1e+06 x[1,5,1] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,5,2]:
   1e+06 x[1,5,2] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,6,0]:
   1e+06 x[1,6,0] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,6,1]:
   1e+06 x[1,6,1] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,6,2]:
   1e+06 x[1,6,2] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,7,0]:
   1e+06 x[1,7,0] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,7,1]:
   1e+06 x[1,7,1] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,7,2]:
   1e+06 x[1,7,2] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,8,0]:
   1e+06 x[1,8,0] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,8,1]:
   1e+06 x[1,8,1] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[1,8,2]:
   1e+06 x[1,8,2] + tau - tau <= 999706.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,4,0]:
   1e+06 x[2,4,0] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,4,1]:
   1e+06 x[2,4,1] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,4,2]:
   1e+06 x[2,4,2] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,5,0]:
   1e+06 x[2,5,0] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,5,1]:
   1e+06 x[2,5,1] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,5,2]:
   1e+06 x[2,5,2] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,6,0]:
   1e+06 x[2,6,0] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,6,1]:
   1e+06 x[2,6,1] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,6,2]:
   1e+06 x[2,6,2] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,7,0]:
   1e+06 x[2,7,0] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,7,1]:
   1e+06 x[2,7,1] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,7,2]:
   1e+06 x[2,7,2] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,8,0]:
   1e+06 x[2,8,0] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,8,1]:
   1e+06 x[2,8,1] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[2,8,2]:
   1e+06 x[2,8,2] + tau - tau <= 999913.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,4,0]:
   1e+06 x[3,4,0] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,4,1]:
   1e+06 x[3,4,1] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,4,2]:
   1e+06 x[3,4,2] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,5,0]:
   1e+06 x[3,5,0] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,5,1]:
   1e+06 x[3,5,1] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,5,2]:
   1e+06 x[3,5,2] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,6,0]:
   1e+06 x[3,6,0] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,6,1]:
   1e+06 x[3,6,1] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,6,2]:
   1e+06 x[3,6,2] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,7,0]:
   1e+06 x[3,7,0] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,7,1]:
   1e+06 x[3,7,1] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,7,2]:
   1e+06 x[3,7,2] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,8,0]:
   1e+06 x[3,8,0] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,8,1]:
   1e+06 x[3,8,1] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[3,8,2]:
   1e+06 x[3,8,2] + tau - tau <= 999670
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,5,0]:
   1e+06 x[4,5,0] + tau - tau <= 999983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,5,1]:
   1e+06 x[4,5,1] + tau - tau <= 999983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,5,2]:
   1e+06 x[4,5,2] + tau - tau <= 999983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,6,0]:
   1e+06 x[4,6,0] + tau - tau <= 999952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,6,1]:
   1e+06 x[4,6,1] + tau - tau <= 999952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,6,2]:
   1e+06 x[4,6,2] + tau - tau <= 999952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,7,0]:
   1e+06 x[4,7,0] + tau - tau <= 999810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,7,1]:
   1e+06 x[4,7,1] + tau - tau <= 999810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,7,2]:
   1e+06 x[4,7,2] + tau - tau <= 999810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,8,0]:
   1e+06 x[4,8,0] + tau - tau <= 999743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,8,1]:
   1e+06 x[4,8,1] + tau - tau <= 999743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[4,8,2]:
   1e+06 x[4,8,2] + tau - tau <= 999743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,4,0]:
   1e+06 x[5,4,0] - tau + tau <= 999983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,4,1]:
   1e+06 x[5,4,1] - tau + tau <= 999983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,4,2]:
   1e+06 x[5,4,2] - tau + tau <= 999983.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,6,0]:
   1e+06 x[5,6,0] + tau - tau <= 999957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,6,1]:
   1e+06 x[5,6,1] + tau - tau <= 999957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,6,2]:
   1e+06 x[5,6,2] + tau - tau <= 999957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,7,0]:
   1e+06 x[5,7,0] + tau - tau <= 999806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,7,1]:
   1e+06 x[5,7,1] + tau - tau <= 999806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,7,2]:
   1e+06 x[5,7,2] + tau - tau <= 999806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,8,0]:
   1e+06 x[5,8,0] + tau - tau <= 999759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,8,1]:
   1e+06 x[5,8,1] + tau - tau <= 999759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[5,8,2]:
   1e+06 x[5,8,2] + tau - tau <= 999759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,4,0]:
   1e+06 x[6,4,0] - tau + tau <= 999952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,4,1]:
   1e+06 x[6,4,1] - tau + tau <= 999952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,4,2]:
   1e+06 x[6,4,2] - tau + tau <= 999952.8
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,5,0]:
   1e+06 x[6,5,0] - tau + tau <= 999957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,5,1]:
   1e+06 x[6,5,1] - tau + tau <= 999957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,5,2]:
   1e+06 x[6,5,2] - tau + tau <= 999957.6
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,7,0]:
   1e+06 x[6,7,0] + tau - tau <= 999845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,7,1]:
   1e+06 x[6,7,1] + tau - tau <= 999845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,7,2]:
   1e+06 x[6,7,2] + tau - tau <= 999845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,8,0]:
   1e+06 x[6,8,0] + tau - tau <= 999773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,8,1]:
   1e+06 x[6,8,1] + tau - tau <= 999773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[6,8,2]:
   1e+06 x[6,8,2] + tau - tau <= 999773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,4,0]:
   1e+06 x[7,4,0] - tau + tau <= 999810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,4,1]:
   1e+06 x[7,4,1] - tau + tau <= 999810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,4,2]:
   1e+06 x[7,4,2] - tau + tau <= 999810.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,5,0]:
   1e+06 x[7,5,0] - tau + tau <= 999806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,5,1]:
   1e+06 x[7,5,1] - tau + tau <= 999806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,5,2]:
   1e+06 x[7,5,2] - tau + tau <= 999806.4
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,6,0]:
   1e+06 x[7,6,0] - tau + tau <= 999845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,6,1]:
   1e+06 x[7,6,1] - tau + tau <= 999845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,6,2]:
   1e+06 x[7,6,2] - tau + tau <= 999845.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,8,0]:
   1e+06 x[7,8,0] + tau - tau <= 999701.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,8,1]:
   1e+06 x[7,8,1] + tau - tau <= 999701.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[7,8,2]:
   1e+06 x[7,8,2] + tau - tau <= 999701.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,4,0]:
   1e+06 x[8,4,0] - tau + tau <= 999743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,4,1]:
   1e+06 x[8,4,1] - tau + tau <= 999743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,4,2]:
   1e+06 x[8,4,2] - tau + tau <= 999743.2
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,5,0]:
   1e+06 x[8,5,0] - tau + tau <= 999759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,5,1]:
   1e+06 x[8,5,1] - tau + tau <= 999759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,5,2]:
   1e+06 x[8,5,2] - tau + tau <= 999759.1
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,6,0]:
   1e+06 x[8,6,0] - tau + tau <= 999773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,6,1]:
   1e+06 x[8,6,1] - tau + tau <= 999773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,6,2]:
   1e+06 x[8,6,2] - tau + tau <= 999773.9
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,7,0]:
   1e+06 x[8,7,0] - tau + tau <= 999701.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,7,1]:
   1e+06 x[8,7,1] - tau + tau <= 999701.3
 Time_window_pizzeria_to_customer_and_customer_to_customer[8,7,2]:
   1e+06 x[8,7,2] - tau + tau <= 999701.3
 Endurance1[4]: tau <= 10317.8
 Endurance1[5]: tau <= 10326.3
 Endurance1[6]: tau <= 10364.9
 Endurance1[7]: tau <= 10436.6
 Endurance1[8]: tau <= 10467.4
 Endurance2: tau <= 10800
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
 x[8,0,2] x_k[0] x_k[1] x_k[2]
Generals
 tau tau tau tau tau tau tau tau tau
End
