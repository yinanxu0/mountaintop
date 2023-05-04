import sys

__all__ = ['main']


def _get_run_args(print_args: bool = True):

    from .parser import get_main_parser

    parser = get_main_parser()
    if len(sys.argv) > 1:
        from argparse import _StoreAction, _StoreTrueAction

        args, unused_args = parser.parse_known_args()

        return args, unused_args
    else:
        parser.print_help()
        exit()


def main():
    """The main entrypoint of the CLI."""
    from mountaintop import bin
    args, unused_args = _get_run_args()
    assert hasattr(args, "main_cli")
    command = getattr(bin, args.main_cli.replace('-', '_'))
    if hasattr(args, "sub_cli"):
        command = getattr(command, args.sub_cli.replace('-', '_'))
    if hasattr(args, "dry_run") and getattr(args, "dry_run"):
        print("DRY RUN mode, args following...")
        print(args)
    else:
        command.run(args, unused_args)


if __name__ == '__main__':
    main()
