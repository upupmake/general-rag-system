import { defineStore } from 'pinia';
import { ref } from 'vue';
import { theme } from 'ant-design-vue';

export const useThemeStore = defineStore('theme', () => {
  const isDark = ref(false);

  localStorage.setItem('theme', 'light');
  document.body.removeAttribute('data-theme');

  return {
    isDark,
    algorithm: () => theme.defaultAlgorithm,
  };
});
