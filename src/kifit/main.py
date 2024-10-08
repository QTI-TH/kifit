from kifit.run import Runner


if __name__ == "__main__":
    # Runner.build().run()

    # Runner.build().print_relative_uncertainties()

    runner = Runner.build()
    runner.generate_all_alphaNP_ll_plots()
    # runner.generate_mphi_alphaNP_plot()
