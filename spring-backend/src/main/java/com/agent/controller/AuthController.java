package com.agent.controller;

import com.agent.model.Medecin;
import com.agent.repository.MedecinRepository;
import com.agent.service.JwtService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/auth")
@CrossOrigin(origins = "*", allowedHeaders = "*")
public class AuthController {

    @Autowired
    private AuthenticationManager authenticationManager;

    @Autowired
    private MedecinRepository medecinRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private JwtService jwtService;

    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody RegisterRequest request) {
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
        medecin.setNom(request.getNom());
        medecin.setPrenom(request.getPrenom());
        medecin.setSpecialite(request.getSpecialite());
        medecin.setTel(request.getTel());
        medecin.setEmail(request.getEmail());
        medecin.setDepartement(request.getDepartement());
        medecin.setUsername(request.getUsername());
        medecin.setPassword(passwordEncoder.encode(request.getPassword()));

        medecinRepository.save(medecin);

        String jwt = jwtService.generateToken(medecin);

        return ResponseEntity.status(HttpStatus.CREATED)
                .body(buildAuthResponse(medecin, jwt));
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

        return ResponseEntity.ok(buildAuthResponse(medecin, jwt));
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
