from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('new_chat', views.new_chat, name='new_chat'),
    path('get_sessions', views.get_sessions, name='get_sessions'),
    path('get_chat/<str:session_id>', views.get_chat, name='get_chat'),
    path('delete_chat/<str:session_id>', views.delete_chat, name='delete_chat'),
    path('send_message', views.send_message, name='send_message'),
    path('dataset_intro', views.dataset_intro, name='dataset_intro'),
    path('regenerate_message', views.regenerate_message, name='regenerate_message'),
    path('message_feedback', views.message_feedback, name='message_feedback'),
]
