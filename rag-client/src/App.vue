<script setup>
import zhCN from 'ant-design-vue/es/locale/zh_CN';
import { useThemeStore } from './stores/theme';

const themeStore = useThemeStore();
</script>

<template>
  <a-config-provider :locale="zhCN" :theme="{ algorithm: themeStore.algorithm() }">
    <router-view />
  </a-config-provider>
</template>

<style scoped>

</style>
