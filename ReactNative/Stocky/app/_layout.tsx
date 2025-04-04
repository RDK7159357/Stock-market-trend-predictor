import { Stack } from "expo-router";
import react from "react";

export default function RootLayout() {
  return <Stack>

    <Stack.Screen name="index" options={{ headerShown: false }} />
    <Stack.Screen name="predict" options={{ headerShown: false }} />

  </Stack>
  
  ;
}
