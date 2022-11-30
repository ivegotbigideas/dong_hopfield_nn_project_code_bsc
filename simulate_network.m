% network information
global number_of_neurons;
number_of_neurons = 2;
global num_stim_vectors;
num_stim_vectors = 6;

% stimulus function
global possible_stimulus_states;
possible_stimulus_states = [1, 1, -1, 1, -1, 1, 1, -1 ,1, -1; -1, 1, -1, -1, -1, 1, 1, 1, -1, -1; 1, 1, -1, -1, -1, 1, 1, 1, -1, -1; -1, 1, -1, 1, 1, 1, 1, 1, -1, -1; -1, 1, -1, -1, -1, 1, 1, 1, -1, -1; -1, 1, -1, -1, -1, 1, -1, 1, -1, -1;];

% initial conditions
u_init = zeros(1, number_of_neurons);
s_init = zeros(1, number_of_neurons^2);
y0 = [u_init, s_init];

% running of the function
% t_span = [0,240];
% func = @simulate;
% [t, y] = ode45(func, t_span, y0);
% plot(t, y(:,11:110));

% testing
disp(simulate(67, [0.4, 0.1, 0, 0.3, 0.3, 0]))

% network state functions
function I = get_I(t)
    global num_stim_vectors;
    global possible_stimulus_states;

    if t<0
        disp("t<0!")
    elseif (0 <= t) && (t < 12)
        I = possible_stimulus_states(1,:);
    elseif (12 <= t) && (t < 24)
        I = possible_stimulus_states(2,:);
    elseif (24 <= t) && (t < 36)
        I = possible_stimulus_states(3,:);
    elseif (36 <= t) && (t < 48)
        I = possible_stimulus_states(4,:);
    elseif (48 <= t) && (t < 60)
        I = possible_stimulus_states(5,:);
    elseif (60 <= t) && (t < 72)
        I = possible_stimulus_states(6,:);
    elseif t == 72
        I = possible_stimulus_states(1,:);
    elseif t > 72
        I = get_I(t-72);
    end
end

% mathematical functions
function sigmoid_value = sigmoid(x)
    if (-1 <= x) && (x <= 1)
        sigmoid_value = x;
    elseif x < -1
        sigmoid_value = -1;
    elseif x > 1
        sigmoid_value = 1;
    end
end

function derivative = dudt(t, u, s, neuron_id)
    global number_of_neurons;

    g = 5;
    a = ones(1, number_of_neurons);
    A = 7;

    term_1 = -u(neuron_id);

    sum = 0;
    for pointer=1:number_of_neurons
        if pointer ~= neuron_id
            connection_strength = sigmoid(s(neuron_id, pointer));
            sum = sum + connection_strength * sigmoid(u(pointer));
        end
    end
    term_2 = g * sum;

    I = get_I(t);
    term_3 = A * I(neuron_id);

    derivative = (1/a(neuron_id)) * (term_1 + term_2 + term_3);
end

function derivative = dsdt(t, u, s, neuron_id_1, neuron_id_2)
    global number_of_neurons;

    H = 1;
    B = 30;

    term_1 = -s(neuron_id_1, neuron_id_2);
    term_2 = H * sigmoid(u(neuron_id_1)) * sigmoid (u(neuron_id_2));

    derivative = 1/B * (term_1 + term_2);
end

function dudt_results = dudt_system(t, u, s)
    global number_of_neurons;

    dudt_results = zeros(1, number_of_neurons);
    for neuron_id_1 = 1:number_of_neurons
        derivative = dudt(t, u, s, neuron_id_1);
        dudt_results(neuron_id_1) = derivative;
    end
end

function dsdt_results = dsdt_system(t, u, s)
    global number_of_neurons;

    dsdt_results = zeros(number_of_neurons, number_of_neurons);
    for neuron_id_1 = 1:number_of_neurons
        for neuron_id_2 = 1:number_of_neurons
            if neuron_id_1 ~= neuron_id_2
                dsdt_results(neuron_id_1, neuron_id_2) = dsdt(t, u, s, neuron_id_1, neuron_id_2);
                if neuron_id_1 == 5 && neuron_id_2 == 4
                    %disp("5,4: " + dsdt_results(neuron_id_1, neuron_id_2))
                elseif neuron_id_1 == 4 && neuron_id_2 == 5
                    %disp("4,5: " + dsdt_results(neuron_id_1, neuron_id_2))
                end
            end
        end
    end
end

function sol = simulate(t, y)
    u = y(1:10);
    s_vector = y(11:110);
    s_vector = transpose(s_vector);

    s = s_vector(1:10);
    s = [s; s_vector(11:20)];
    s = [s; s_vector(21:30)];
    s = [s; s_vector(31:40)];
    s = [s; s_vector(51:60)];
    s = [s; s_vector(61:70)];
    s = [s; s_vector(71:80)];
    s = [s; s_vector(81:90)];
    s = [s; s_vector(91:100)];
    s = [s; s_vector(91:100)];

    dudt_results = dudt_system(t, u, s);
    dsdt_results = dsdt_system(t, u, s);

    dsdt_vector_results(1:10) = dsdt_results(1, 1:10);
    dsdt_vector_results(11:20) = dsdt_results(2, 1:10);
    dsdt_vector_results(21:30) = dsdt_results(3, 1:10);
    dsdt_vector_results(31:40) = dsdt_results(4, 1:10);
    dsdt_vector_results(41:50) = dsdt_results(5, 1:10);
    dsdt_vector_results(51:60) = dsdt_results(6, 1:10);
    dsdt_vector_results(61:70) = dsdt_results(7, 1:10);
    dsdt_vector_results(71:80) = dsdt_results(8, 1:10);
    dsdt_vector_results(81:90) = dsdt_results(9, 1:10);
    dsdt_vector_results(91:100) = dsdt_results(10, 1:10);

    sol = transpose([dudt_results, dsdt_vector_results]);
end
