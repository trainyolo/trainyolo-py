import sys
from argparse import ArgumentParser
import argcomplete

class Options:

    def __init__(self):

        self.parser = ArgumentParser(
            description="Cli for trainYOLO."
        )

        subparsers = self.parser.add_subparsers(dest="command")

        # authenticate
        subparsers.add_parser("authenticate", help="Store apikey for authentication.")

        # project
        project = subparsers.add_parser("project", help="project related functions")
        project_action = project.add_subparsers(dest="action")

        ## project create
        project_create = project_action.add_parser("create", help="Create a new project")
        project_create.add_argument(
            "name",
            type=str,
            help="Name of project"
        )
        project_create.add_argument(
            "--type",
            type=str,
            default="BBOX",
            help="project type: BBOX or INSTANCE_SEGMENTATION"
        )
        project_create.add_argument(
            "--categories",
            type=str,
            default="object",
            help="list of categories, use comma between categories, eg: cat_1,cat_2"
        )

        ## project pull
        project_pull = project_action.add_parser("pull", help="Pull samples from project")
        project_pull.add_argument(
            "project",
            type=str,
            help="Name of project"
        )
        project_pull.add_argument(
            "--path",
            type=str,
            default='./',
            help="Path to image folder"
        )
        project_pull.add_argument(
            "--format",
            type=str,
            default='yolov5',
            help="Output format"
        )

        ## project push
        project_push = project_action.add_parser("push", help="Push images to project")
        project_push.add_argument(
            "project",
            type=str,
            help="Name of project"
        )
        project_push.add_argument(
            "path",
            type=str,
            help="Path to image folder"
        )

        ## project add_model
        project_add_model = project_action.add_parser("add_model", help="add model to project")
        project_add_model.add_argument(
            "project",
            type=str,
            help="Name of project"
        )

        project_add_model.add_argument(
            "--type",
            type=str,
            default='yolov5',
            help="Model type, eg yolov5/yolov8/yolov8-seg/yolov8-pose"
        )

        project_add_model.add_argument(
            "--run-location",
            type=str,
            help="Run location, typically ./runs"
        )
        project_add_model.add_argument(
            "--run",
            type=str,
            help="Run, eg exp (optional, if omitted will use latest run)"
        )
        project_add_model.add_argument(
            "--conf",
            type=float,
            default=None,
            help="conf value, if ommited will use the one that maximizes f1 score"
        )
        project_add_model.add_argument(
            "--iou",
            type=float,
            default=.45,
            help="Nms threshold value"
        )

        # version
        subparsers.add_parser("version", help="Print current version number")

        argcomplete.autocomplete(self.parser)

    def parse_args(self):
        args = self.parser.parse_args()

        if not args.command:
            self.parser.print_help()
            sys.exit()

        return args, self.parser