Network benchmark dataset simulator with Django WebApp

Build instructions:
1.) Create a virtual environment (You can also use a newer Python version):
virtualenv -p python3.6 .

2.) Activate the virtual environment:
source bin/activate

3.) Install the requirements
pip install -r requirements.txt

4.) Change into the src folder
cd src

5.) Start the Django project
# this only needed to be done once in the beginning, no need to do that: django-admin.py startproject simulator .
python manage.py migrate

6.) Create the superuser
python manage.py createsuperuser

7.) Run the server
python manage.py runserver


Note: Whenever you are making changes to the code, you need to run this:
python manage.py makemigrations
python manage.py migrate

