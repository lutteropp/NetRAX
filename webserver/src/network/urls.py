from django.urls import path

from .views import (
    NetworkCreateView,
    NetworkDetailView,
    NetworkListView,
    NetworkUpdateView,
    NetworkDeleteView
    )

app_name = 'network'
urlpatterns = [
    path('', NetworkListView.as_view(), name='network-list'),
    #path('', network_list_view, name='network-list'),
    path('create/', NetworkCreateView.as_view(), name='network-create'),
	path('<int:pk>', NetworkDetailView.as_view(), name='network-detail'),
    path('<int:pk>/update/', NetworkUpdateView.as_view(), name='network-update'),
    path('<int:pk>/delete/', NetworkDeleteView.as_view(), name='network-delete'),
]
