.. highlight:: shell

Contributing
------------

Thank you for considering contributing to the PLISM robustness benchmark project!
This section provides instructions on how to set up the project locally and how to
contribute to the codebase.


Installation
~~~~~~~~~~~~

To get started, you can download the source code from the `Github repository`_ by
cloning it:

You can clone the repository:

.. code-block:: console

    $ git clone git@github.com:owkin/plism-benchmark.git

Once you have a copy of the source code, we recommend installing the latest version of
`poetry`_:

.. code-block:: console

    $ make install-poetry

Next, create a Python environment using your preferred management system (``conda``,
``pip``, ...). If you don't have a preferred system, ``poetry`` will automatically
create a new environment in ``.venv/`` when you run ``make config``. If you're using
your own environment, make sure to activate it. To activate ``poetry``'s environment,
you can run: ``poetry shell``. Please configure ``poetry`` by running:

.. code-block:: console

    $ make config


Once that's done, and if it has not been generated yet,
you must generate the ``poetry.lock`` file by running:

.. code-block:: console

    $ make lock

To install all required dependencies, you can run the following command:

.. code-block:: console

    $ make install-all

.. _Github repository: https://github.com/owkin/plism-benchmark
.. _poetry: https://python-poetry.org/docs/


Pre-commit
~~~~~~~~~~

You can run all the aforementioned styling checks manually as described.
However, we encourage you to use `pre-commit hooks <https://pre-commit.com/>`_
instead to automatically run ``ruff`` and ``mypy``.
This can be done by running :

.. code-block:: console

    $ pre-commit install

from the root of the ``plism-benchmark`` repository. Now all of
the styling checks will be run each time you commit changes without your
needing to run each one manually. In addition, using ``pre-commit`` will also
allow you to more easily remain up-to-date with code checks as they evolve.

If you don’t want to use ``pre-commit`` as part of your workflow, you can
still use it to run its checks with:

.. code-block:: console

    $ make pre-commit-checks

without needing to have done ``pre-commit install`` beforehand.


Guidelines
~~~~~~~~~~

To contribute to the PLISM robustness benchmark project, follow these steps:

    1. Create a new branch for your changes.
    2. Make your changes and commit them with clear commit messages.
    3. Push your changes to your branch.

When opening a pull request, make sure to include a clear description of your changes
and why they are necessary.


Testing
~~~~~~~

The PLISM robustness benchmark project uses  `pytest <https://docs.pytest.org/>`_
for testing. To run the tests, simply run:

.. code-block:: console

    $ make test-all


Make sure that all tests pass before submitting a pull request.


Documentation
~~~~~~~~~~~~~

The PLISM robustness benchmark project uses `Sphinx <https://www.sphinx-doc.org/>`_
for documentation. To build the documentation, run:

.. code-block:: console

    $ make docs

The documentation will be built in the ``docs/_build/`` directory.


New dependencies
~~~~~~~~~~~~~~~~

If or when you add additional dependencies to your project, you can use ``poetry``
in the following manner:

.. code-block:: console

    $ poetry add xformers


If you already have a ``requirements.txt`` file with your dependencies, you can inject
them using ``poetry`` with the command:

.. code-block:: console

    $ cat requirements.txt | xargs poetry add


If your project requires dependencies that can't be installed using pip, make sure to
add the corresponding installation commands to the ``Makefile`` under the
``make install`` section like this:

.. code-block:: Makefile

    install: clean
        conda install <conda-specific-dependency>  # Example of dependency only installed with conda
        curl <bash-specific-dependency> | sh  # Example of dependency only installed with bash
        poetry install

You can also add a library located in a git repository, the minimum information you
need to specify is the location of the repository with the git key, and if necessary
the branch from which the library is to be installed. By default ``poetry`` will revert
to the master branch. You can do using the following command:

.. code-block:: console

    $ poetry add "https://github.com/org/mypackage.git#branch=my_branch"


Useful tip
~~~~~~~~~~

The repository comes with a preconfigured ``Makefile`` encapsulating numerous
useful commands. To check them out, run the command:

.. code-block:: console

    $ make help
