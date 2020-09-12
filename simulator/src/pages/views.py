from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def home_view(request, *args, **kwargs):
	return render(request, "home.html", {})

def add_view(request, *args, **kwargs):
	return render(request, "add.html", {})

def dataset_view(request, *args, **kwargs):
	my_context = {
		"my_text": "This is a test"
	}
	return render(request, "dataset.html", my_context)