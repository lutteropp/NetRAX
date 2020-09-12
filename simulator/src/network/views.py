from django.urls import reverse
from django.shortcuts import render, redirect, get_object_or_404

from django.views.generic import (
	CreateView,
	DetailView,
	ListView,
	UpdateView,
	DeleteView,
	View
	)

from .forms import NetworkModelForm
from .models import Network
# Create your views here.

class NetworkObjectMixin(object):
	model = Network
	lookup = 'pk'

	def get_object(self):
		id_ = self.kwargs.get(self.lookup)
		return get_object_or_404(self.model, id=id_)

class NetworkRawCreateView(View):
	template_name = 'network/network_create.html'

	def get(self, request, *args, **kwargs):
		form = NetworkModelForm()
		context = {"form": form}
		return render(request, self.template_name, context)

	def post(self, request, *args, **kwargs):
		form = NetworkModelForm(request.POST)
		if form.is_valid():
			form.save()
			form = NetworkModelForm()
		context = {"form": form}
		return render(request, self.template_name, context)


class NetworkListView(ListView):
	template_name = 'network/network_list.html'
	queryset = Network.objects.all()


class NetworkDetailView(NetworkObjectMixin, DetailView):
	template_name = 'network/network_detail.html'
	queryset = Network.objects.all()


class NetworkCreateView(CreateView):
	template_name = 'network/network_create.html'
	form_class = NetworkModelForm
	queryset = Network.objects.all()


class NetworkUpdateView(NetworkObjectMixin, UpdateView):
	template_name = 'network/network_create.html'
	queryset = Network.objects.all()


class NetworkDeleteView(NetworkObjectMixin, DeleteView):
	template_name = 'network/network_delete.html'

	def get_success_url(self):
		return reverse('network:network-list')