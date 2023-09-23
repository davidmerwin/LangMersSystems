#include <gtk/gtk.h>

void on_btn_login_clicked(GtkButton *button, gpointer user_data) {
    GtkEntry *username_field = GTK_ENTRY(user_data);
    GtkEntry *password_field = GTK_ENTRY(g_object_get_data(G_OBJECT(button), "password_field"));

    const gchar *username = gtk_entry_get_text(username_field);
    const gchar *password = gtk_entry_get_text(password_field);

    if (g_strcmp0(username, "correct-username") == 0 && g_strcmp0(password, "correct-password") == 0) {
        // Log in
    } else {
        // Show error message
    }
}

int main(int argc, char *argv[]) {
    GtkBuilder *builder;
    GtkWidget *window;
    GtkEntry *username_field;
    GtkEntry *password_field;

    gtk_init(&argc, &argv);

    builder = gtk_builder_new();
    gtk_builder_add_from_file(builder, "login.glade", NULL);

    window = GTK_WIDGET(gtk_builder_get_object(builder, "window_main"));
    if (!GTK_IS_WIDGET(window)) {
        g_error("Failed to load window_main from login.glade");
        return 1;
    }
    gtk_widget_show_all(window);

    username_field = GTK_ENTRY(gtk_builder_get_object(builder, "username_field"));
    password_field = GTK_ENTRY(gtk_builder_get_object(builder, "password_field"));

    GtkButton *btn_login = GTK_BUTTON(gtk_builder_get_object(builder, "btn_login"));
    if (!GTK_IS_BUTTON(btn_login)) {
        g_error("Failed to load btn_login from login.glade");
        return 1;
    }
    g_object_set_data(G_OBJECT(btn_login), "password_field", password_field);

    g_signal_connect(btn_login, "clicked", G_CALLBACK(on_btn_login_clicked), username_field);

    gtk_main();
    return 0;
}
