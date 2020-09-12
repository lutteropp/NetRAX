Network benchmark dataset simulator with Django WebApp

Build instructions:
1.) Create a virtual environment:
virtualenv -p python3.8 .

2.) Activate the virtual environment:
source bin/activate

3.) Install the requirements
pip install -r requirements.txt

4.) Change into the src folder
cd src

5.) Start the Django project
django-admin.py startproject simulator .
python manage.py migrate

6.) Create the superuser
python manage.py createsuperuser

7.) Run the server
python manage.py runserver


Note: Whenever you are making changes to the code, you need to run this:
python manage.py makemigrations
python manage.py migrate

