// driver.c
#include <stdio.h>
#include <stdint.h>

extern int32_t entry_main(void);

int main(void) {
  int32_t result = entry_main();
  printf("%d\n", (int)result);
  return 0;
}
