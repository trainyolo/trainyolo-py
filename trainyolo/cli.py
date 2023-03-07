from trainyolo.cli_options import Options
from trainyolo import cli_functions
import getpass
from trainyolo import __version__


def main():
    
    args, parser = Options().parse_args()
    exec_command(args, parser)

def exec_command(args, parser):

    if args.command == "authenticate":
        apikey = getpass.getpass(prompt="API key: ", stream=None)
        apikey = apikey.strip()
        if apikey == "":
            print("No API key entered. Consult the trainyolo documentation on how to get your API key.")
            return
        else:
            cli_functions.authenticate(apikey)

    if args.command == "version":
        print(__version__)

    if args.command == "project": 
        if args.action == "create":
            cli_functions.create_project(args.name, args.type, args.categories)
        elif args.action == "pull":
            cli_functions.pull_project(args.project, args.path, args.format)
        elif args.action == "push":
            cli_functions.push_to_project(args.project, args.path)
        elif args.action == "add_model":
            cli_functions.add_model(args.project, args.type, args.run_location, args.run, args.conf, args.iou)

        
if __name__ == "__main__":
    main()