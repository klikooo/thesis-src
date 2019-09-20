from plots.cnn.plot_ascad_rd_fixed import plot_ascad

if __name__ == "__main__":
    ########################
    # PLOT WITH EQUAL AXES #
    ########################
    # limits_x = [[-2, 1000]] * 5
    # limits_y = [[-5, 128]] * 5
    # plot_ascad(hw=False, desync=50, noise_level=0.0, x_limits=limits_x, y_limits=limits_y,
    #            show=False, file_extension="equal")
    #

    limits_x = [[0, 10000]] * 5
    limits_y = [[0, 250]] * 5
    plot_ascad(hw=True, desync=0, noise_level=0.0, x_limits=limits_x, y_limits=limits_y,
               show=False, file_extension="fitting", unmask=False,
               file_path="/media/rico/Data/TU/thesis/report/img/cnn/ascad_masked")


    ###############################
    # PLOT WITH GOOD FITTING AXES #
    # ###############################
    # limits_x = [[-2, 400], [-2, 1500], [-2, 400], [-2, 400], [-2, 400], [-2, 400], [-2, 400], [-2, 400], [-2, 400]]
    # limits_y = [[-5, 128], [-5, 128], [-5, 128], [-5, 128], [-5, 128], [-5, 128], [-5, 128], [-5, 128], [-5, 128]]
    # plot_ascad(0, limits_x, limits_y, show=False, file_extension="fitting")
