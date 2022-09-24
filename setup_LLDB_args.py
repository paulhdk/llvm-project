import lldb
import shlex

with open(".compile_args", "r") as f:
    target = lldb.debugger.GetTargetAtIndex(0)

    launch_info = target.GetLaunchInfo()

    args = shlex.split(f.read())

    launch_info.SetArguments(args, False)

    target.SetLaunchInfo(launch_info)
