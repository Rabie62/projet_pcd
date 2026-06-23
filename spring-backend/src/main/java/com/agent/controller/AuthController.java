package com.agent.controller;

import com.agent.model.Medecin;
import com.agent.repository.MedecinRepository;
import com.agent.service.JwtService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.ResponseCookie;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/auth")
public class AuthController {

    private final AuthenticationManager authenticationManager;
    private final MedecinRepository medecinRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;

    public AuthController(AuthenticationManager authenticationManager,
                          MedecinRepository medecinRepository,
                          PasswordEncoder passwordEncoder,
                          JwtService jwtService) {
        this.authenticationManager = authenticationManager;
        this.medecinRepository = medecinRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtService = jwtService;
    }

    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody RegisterRequest request) {
        // Input validation
        String validationError = validateRegistration(request);
        if (validationError != null) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(Map.of("message", validationError));
        }

        if (medecinRepository.existsByUsername(request.getUsername())) {
            return ResponseEntity.status(HttpStatus.CONFLICT)
                    .body(Map.of("message", "Le nom d'utilisateur est déjà pris."));
        }

        if (request.getEmail() != null && !request.getEmail().isEmpty()
            && medecinRepository.existsByEmail(request.getEmail())) {
            return ResponseEntity.status(HttpStatus.CONFLICT)
                    .body(Map.of("message", "Cet e-mail est déjà associé à un compte."));
        }

        Medecin medecin = new Medecin();
        medecin.setNom(request.getNom().trim());
        medecin.setPrenom(request.getPrenom().trim());
        medecin.setSpecialite(request.getSpecialite());
        medecin.setTel(request.getTel());
        medecin.setEmail(request.getEmail() != null ? request.getEmail().trim().toLowerCase() : null);
        medecin.setDepartement(request.getDepartement());
        medecin.setUsername(request.getUsername().trim().toLowerCase());
        medecin.setPassword(passwordEncoder.encode(request.getPassword()));

        medecinRepository.save(medecin);

        String jwt = jwtService.generateToken(medecin);

        ResponseCookie cookie = ResponseCookie.from("token", jwt)
                .httpOnly(true)
                .secure(false)
                .sameSite("Lax")
                .path("/")
                .maxAge(86400)
                .build();

        return ResponseEntity.status(HttpStatus.CREATED)
                .header("Set-Cookie", cookie.toString())
                .body(buildAuthResponse(medecin, jwt));
    }

    /**
     * Validates registration input. Returns an error message if invalid, null if valid.
     */
    private String validateRegistration(RegisterRequest request) {
        if (request.getNom() == null || request.getNom().trim().isEmpty()) {
            return "Le nom est obligatoire.";
        }
        if (request.getPrenom() == null || request.getPrenom().trim().isEmpty()) {
            return "Le prénom est obligatoire.";
        }
        if (request.getUsername() == null || request.getUsername().trim().isEmpty()) {
            return "Le nom d'utilisateur est obligatoire.";
        }
        if (request.getUsername().trim().length() < 3) {
            return "Le nom d'utilisateur doit contenir au moins 3 caractères.";
        }
        if (request.getUsername().trim().length() > 50) {
            return "Le nom d'utilisateur ne peut pas dépasser 50 caractères.";
        }
        if (!request.getUsername().trim().matches("^[a-zA-Z0-9._-]+$")) {
            return "Le nom d'utilisateur ne peut contenir que des lettres, chiffres, points, tirets et underscores.";
        }
        if (request.getPassword() == null || request.getPassword().isEmpty()) {
            return "Le mot de passe est obligatoire.";
        }
        if (request.getPassword().length() < 8) {
            return "Le mot de passe doit contenir au moins 8 caractères.";
        }
        if (request.getPassword().length() > 128) {
            return "Le mot de passe ne peut pas dépasser 128 caractères.";
        }
        // Check password complexity: at least one uppercase, one lowercase, one digit
        if (!request.getPassword().matches(".*[A-Z].*")) {
            return "Le mot de passe doit contenir au moins une lettre majuscule.";
        }
        if (!request.getPassword().matches(".*[a-z].*")) {
            return "Le mot de passe doit contenir au moins une lettre minuscule.";
        }
        if (!request.getPassword().matches(".*[0-9].*")) {
            return "Le mot de passe doit contenir au moins un chiffre.";
        }
        if (request.getEmail() != null && !request.getEmail().isEmpty()) {
            if (!request.getEmail().trim().matches("^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$")) {
                return "L'adresse e-mail n'est pas valide.";
            }
        }
        return null;
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest request) {
        try {
            authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(
                            request.getUsername(),
                            request.getPassword()
                    )
            );
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("message", "Identifiants de connexion incorrects."));
        }

        Medecin medecin = medecinRepository.findByUsername(request.getUsername())
                .orElseThrow();

        String jwt = jwtService.generateToken(medecin);

        ResponseCookie cookie = ResponseCookie.from("token", jwt)
                .httpOnly(true)
                .secure(false)
                .sameSite("Lax")
                .path("/")
                .maxAge(86400)
                .build();

        return ResponseEntity.ok()
                .header("Set-Cookie", cookie.toString())
                .body(buildAuthResponse(medecin, jwt));
    }

    @PostMapping("/logout")
    public ResponseEntity<?> logout() {
        // With httpOnly cookies, the client can't clear the cookie directly.
        // The server sets an empty cookie to clear it.
        ResponseCookie cookie = ResponseCookie.from("token", "")
                .httpOnly(true)
                .secure(true)
                .sameSite("Strict")
                .path("/")
                .maxAge(0)
                .build();

        return ResponseEntity.ok()
                .header("Set-Cookie", cookie.toString())
                .body(Map.of("message", "Déconnexion réussie."));
    }

    private Map<String, Object> buildAuthResponse(Medecin medecin, String token) {
        Map<String, Object> response = new HashMap<>();
        response.put("token", token);
        response.put("id", medecin.getId());
        response.put("nom", medecin.getNom());
        response.put("prenom", medecin.getPrenom());
        response.put("email", medecin.getEmail());
        response.put("specialite", medecin.getSpecialite());
        response.put("departement", medecin.getDepartement());
        response.put("username", medecin.getUsername());
        return response;
    }

    // --- Helper classes for requests ---

    public static class RegisterRequest {
        private String nom;
        private String prenom;
        private String specialite;
        private String tel;
        private String email;
        private String departement;
        private String username;
        private String password;

        public String getNom() { return nom; }
        public void setNom(String nom) { this.nom = nom; }

        public String getPrenom() { return prenom; }
        public void setPrenom(String prenom) { this.prenom = prenom; }

        public String getSpecialite() { return specialite; }
        public void setSpecialite(String specialite) { this.specialite = specialite; }

        public String getTel() { return tel; }
        public void setTel(String tel) { this.tel = tel; }

        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }

        public String getDepartement() { return departement; }
        public void setDepartement(String departement) { this.departement = departement; }

        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }

        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
    }

    public static class LoginRequest {
        private String username;
        private String password;

        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }

        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
    }
}
