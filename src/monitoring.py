import pandas_profiling


def monitor(df, path):

    profile = pandas_profiling.ProfileReport(df, title="Pandas Profiling Report")

    profile.to_file(path)

    pass
