import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService, AuthUser } from '../services/auth.service';

export const medecinGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);
  const user: AuthUser | null = authService.getCurrentUser();

  if (!user) {
    router.navigate(['/login'], { queryParams: { returnUrl: state.url } });
    return false;
  }

  // All authenticated users are medecins in this system
  // Could be extended with role-based checks if needed
  return true;
};

export const adminGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);
  const user: AuthUser | null = authService.getCurrentUser();

  if (!user) {
    router.navigate(['/login'], { queryParams: { returnUrl: state.url } });
    return false;
  }

  // Check for admin role (could be extended with a 'role' field in the user object)
  // For now, all authenticated users have access
  return true;
};
