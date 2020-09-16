.. Scicrop documentation master file, created by
   sphinx-quickstart on Wed Sep 16 08:44:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



.. image:: https://agevolution.canalrural.com.br/wp-content/uploads/2019/06/Scicrop.jpg
   :target: https://scicrop.com/?lang=en
   :align: center
   :width: 350


My Scicrop job application!
===================================


Check out the GitHub repository:

.. image:: /images/hiclipart.com.png
   :target: https://github.com/feippolito/jobs-datascience
   :width: 75
   :align: center

|
|

This project is a solution to a multi-class supervised learning task. The repository
is structured in 5 notebooks in which we explore the data and try different approaches.

1. FirstAnalysis
2. Missing Values
3. Machine Learning
4. Optimize
5. Validate

After we have the best solution we implement them in python scrips, there are 3 main scripts:

1. train_test_split.py
2. create_models.py
3. predict.py

This is a complete data pipeline - from preparing the data, training, optimizing,
saving the models and predicting target data. This pipeline can be ran using
the make command:

.. code-block:: bash
  :linenos:

  make train_predict

The data pipeline can also be ran using a docker container!

.. code-block:: bash
  :linenos:

  make docker

Here we build the docker image, mount the host :code:`results/` directory to the cointainer's
directory and run the pipeline, the prediction output is saved in the host.


📓 First Analysis
=======================================

.. toctree::
   :maxdepth: 1

   notebooks/1.FirstAnalysis

📓 Missing Values
=======================================

.. toctree::
   :maxdepth: 1

   notebooks/2.MissingValues

📓 Machine Learning
=======================================

.. toctree::
   :maxdepth: 1

   notebooks/3.MachineLearning

📓 Optimization
=======================================

.. toctree::
   :maxdepth: 1

   notebooks/4.Optimize

📓 Validation
=======================================

.. toctree::
   :maxdepth: 1

   notebooks/5.Validate

🖥️ Code Documentation
=======================================

.. toctree::
   :maxdepth: 2

   lib
