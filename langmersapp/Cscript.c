#include <stdio.h>
#include <string.h>

void provide_word_details(char *word) {
    printf("Looked up details for the word: %s\n", word);
}

int main() {
    char recognized_word[] = "table";

    provide_word_details(recognized_word);

    return 0;
}
