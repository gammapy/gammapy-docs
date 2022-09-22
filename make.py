#!/usr/bin/env python
import subprocess
import click

# Global config
config = {}


def run(cmd):
    print('Execute command: ', cmd)
    if not config['dry']:
        subprocess.call(cmd, shell=True)


@click.group()
@click.option('--dry', is_flag=True,
              help='Dry run (just print, don\'t execute commands)')
def cli(dry):
    """Gammapy docs build & deploy tool"""
    config['dry'] = dry


@cli.command('htmlcopy')
@click.argument('version')
def cli_htmlcopy(version):
    """Copy HTML from build to docs folder"""
    cmd = f'cp -r build/{version}/gammapy/docs/_build/html/* docs/{version}/'
    run(cmd)


if __name__ == '__main__':
    cli()

"""
To check the repo size
(from https://stackoverflow.com/questions/8646517/see-the-size-of-a-github-repo-before-cloning-it)
curl https://api.github.com/repos/gammapy/gammapy-docs 2> /dev/null | grep size | tr -dc '[:digit:]'
"""
