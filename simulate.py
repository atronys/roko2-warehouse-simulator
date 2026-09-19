from Simulator import Simulator

simulation_done = False
sim_time = 20000

main_simulator = Simulator(sim_time, None)

while not simulation_done:
    main_simulator.measure()
    main_simulator.calculate_control()
    
    if not main_simulator.move():
        break
    
    main_simulator.accumulate_data()
    
    # Проверяем завершение сегмента
    if main_simulator.check_target():
        break
    
    simulation_done = main_simulator.check_simulation_done()

main_simulator.plot_main_data()